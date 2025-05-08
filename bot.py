import os
import time
import logging
from dotenv import load_dotenv
from atproto import Client, models
from atproto.exceptions import AtProtocolError
import google.generativeai as genai
import re # Import regular expressions
# Need BytesIO if Gemini returns image bytes? (Might not be needed if data is already bytes)
# from io import BytesIO 

# Import the specific Params model
from atproto_client.models.app.bsky.notification.list_notifications import Params as ListNotificationsParams
# Import the specific Params model for get_post_thread
from atproto_client.models.app.bsky.feed.get_post_thread import Params as GetPostThreadParams
# Import the specific Params model for get_posts
from atproto_client.models.app.bsky.feed.get_posts import Params as GetPostsParams
# Import Facet and Embed models
from atproto import models as at_models 

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

BLUESKY_HANDLE = os.getenv("BLUESKY_HANDLE")
BLUESKY_PASSWORD = os.getenv("BLUESKY_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Environment Variables
GEMINI_MODEL_NAME = "models/gemini-2.0-flash-latest" # Changed based on ListModels output

# Constants
BOT_SYSTEM_INSTRUCTION = "You are a helpful assistant on the Bluesky social network. Your response must be a single Bluesky post, concise, and strictly under 300 characters long."
MENTION_CHECK_INTERVAL_SECONDS = 15 # Check for new mentions every 15 seconds (was 60)
MAX_THREAD_DEPTH_FOR_CONTEXT = 15 # How many parent posts to fetch for context
NOTIFICATION_FETCH_LIMIT = 100 # How many notifications to fetch (was 25)

# Global variables
bsky_client: Client | None = None
gemini_model: genai.GenerativeModel | None = None
processed_uris_this_run: set[str] = set() # Track URIs processed in this run to handle is_read lag

def initialize_bluesky_client() -> Client | None:
    """Initializes and logs in the Bluesky client."""
    if not (BLUESKY_HANDLE and BLUESKY_PASSWORD):
        logging.error("Bluesky handle or password not found in environment variables.")
        return None
    
    client = Client()
    try:
        profile = client.login(BLUESKY_HANDLE, BLUESKY_PASSWORD)
        logging.info(f"Successfully logged in to Bluesky as {profile.handle}")
        return client
    except AtProtocolError as e:
        logging.error(f"Bluesky login failed: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during Bluesky login: {e}")
        return None

def initialize_gemini_model() -> genai.GenerativeModel | None:
    """Initializes the Gemini generative model."""
    if not GEMINI_API_KEY:
        logging.error("Gemini API key not found in environment variables.")
        return None
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL_NAME, 
            system_instruction=BOT_SYSTEM_INSTRUCTION,
            safety_settings=[ 
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
        )
        logging.info(f"Successfully initialized Gemini model with: {GEMINI_MODEL_NAME}")
        return model
    except Exception as e:
        logging.error(f"Failed to initialize Gemini model: {e}", exc_info=True)
        return None

def format_thread_for_gemini(thread_view: models.AppBskyFeedDefs.ThreadViewPost, own_handle: str) -> str | None:
    """
    Formats the thread leading up to and including the mentioned_post into a string for Gemini.
    `thread_view` is the ThreadViewPost for the post that contains the mention.
    """
    history = []
    current_view = thread_view

    while current_view:
        if isinstance(current_view, models.AppBskyFeedDefs.ThreadViewPost) and current_view.post:
            post_record = current_view.post.record
            if isinstance(post_record, models.AppBskyFeedPost.Record) and hasattr(post_record, 'text'):
                author_display_name = current_view.post.author.display_name or current_view.post.author.handle
                text = post_record.text
                history.append(f"{author_display_name} (@{current_view.post.author.handle}): {text}")
        elif isinstance(current_view, (models.AppBskyFeedDefs.NotFoundPost, models.AppBskyFeedDefs.BlockedPost)):
            logging.warning(f"Encountered NotFoundPost or BlockedPost while traversing thread parent: {current_view}")
            break 
        
        if hasattr(current_view, 'parent') and current_view.parent:
            current_view = current_view.parent
        else:
            break

    history.reverse() 
    
    if not history:
        logging.warning("Could not construct any context from the thread.")
        if isinstance(thread_view.post.record, models.AppBskyFeedPost.Record) and hasattr(thread_view.post.record, 'text'):
            author_display_name = thread_view.post.author.display_name or thread_view.post.author.handle
            return f"{author_display_name} (@{thread_view.post.author.handle}): {thread_view.post.record.text}"
        return None
        
    return "\\\\n\\\\n".join(history)

def process_mention(notification: at_models.AppBskyNotificationListNotifications.Notification, gemini_model_ref: genai.GenerativeModel):
    """Processes a single mention/reply notification."""
    global bsky_client, processed_uris_this_run # Ensure globals are accessible
    if not bsky_client:
        logging.error("Bluesky client not initialized in process_mention. Cannot process mention.")
        return

    mentioned_post_uri = notification.uri
    # Mark as seen for this run *before* any processing attempts to prevent loops if is_read lags
    processed_uris_this_run.add(mentioned_post_uri)
    
    logging.info(f"Processing mention/reply in post: {mentioned_post_uri}")

    try:
        params = GetPostThreadParams(uri=mentioned_post_uri, depth=MAX_THREAD_DEPTH_FOR_CONTEXT)
        thread_view_response = bsky_client.app.bsky.feed.get_post_thread(params=params)
        
        if not isinstance(thread_view_response.thread, at_models.AppBskyFeedDefs.ThreadViewPost):
            logging.warning(f"Could not fetch thread or thread is not a ThreadViewPost for {mentioned_post_uri}. Type: {type(thread_view_response.thread)}")
            return

        thread_view_of_mentioned_post = thread_view_response.thread
        target_post = thread_view_of_mentioned_post.post
        if not target_post:
            logging.warning(f"Thread view for {mentioned_post_uri} does not contain a post.")
            return
            
        # --- Reason-Specific Logic ---
        if notification.reason == 'mention':
            logging.info(f"[Mention Check] Processing mention in {target_post.uri}")
            # Check for existing replies by the bot under the mentioned post (target_post)
            if thread_view_of_mentioned_post.replies:
                 for reply_in_thread in thread_view_of_mentioned_post.replies:
                    if reply_in_thread.post and reply_in_thread.post.author and reply_in_thread.post.author.handle == BLUESKY_HANDLE:
                        logging.info(f"[DUPE CHECK MENTION] Found pre-existing bot reply {reply_in_thread.post.uri} to mentioned post {target_post.uri}. Skipping.")
                        return
            # If no duplicate found, fall through to generate context and reply...
            logging.info(f"[Mention Check] No duplicate bot reply found for mention {target_post.uri}. Proceeding.")

        elif notification.reason == 'reply':
            logging.info(f"[Reply Check] Processing reply notification for {target_post.uri}")
            post_record = target_post.record
            # Check if the target post is a valid reply with parent info
            if isinstance(post_record, at_models.AppBskyFeedPost.Record) and post_record.reply and post_record.reply.parent:
                parent_ref = post_record.reply.parent
                logging.info(f"[Reply Check] Post {target_post.uri} replies to parent URI: {parent_ref.uri}. Fetching parent...")
                try:
                    get_parent_params = GetPostsParams(uris=[parent_ref.uri])
                    parent_post_response = bsky_client.app.bsky.feed.get_posts(params=get_parent_params)
                    if parent_post_response and parent_post_response.posts and len(parent_post_response.posts) == 1:
                        immediate_parent_post = parent_post_response.posts[0]
                        logging.info(f"[Reply Check] Fetched immediate parent post. Author: {immediate_parent_post.author.handle}")

                        # **REVISED LOGIC**: Only proceed if the immediate parent IS the bot.
                        if immediate_parent_post.author.handle == BLUESKY_HANDLE:
                            logging.info(f"[Reply Check] Immediate parent is the bot. Checking for duplicate replies under {target_post.uri}...")
                            # Check for existing replies by the bot under the *triggering* post (target_post)
                            if thread_view_of_mentioned_post.replies:
                                 for reply_to_users_reply in thread_view_of_mentioned_post.replies:
                                    if reply_to_users_reply.post and reply_to_users_reply.post.author and \
                                       reply_to_users_reply.post.author.handle == BLUESKY_HANDLE:
                                        logging.info(f"[DUPE CHECK REPLY] Found pre-existing bot reply {reply_to_users_reply.post.uri} under user's reply {target_post.uri}. Skipping.")
                                        return
                            # If no duplicate found, fall through to generate context and reply...
                            logging.info(f"[Reply Check] No duplicate bot reply found under {target_post.uri}. Proceeding.")
                        else:
                            # Parent is another user, ignore this reply.
                            logging.info(f"[IGNORE USER-TO-USER REPLY] Notification {notification.uri} is a reply to another user ({immediate_parent_post.author.handle}), not the bot. Ignoring.")
                            return
                    else:
                        logging.warning(f"[Reply Check] Failed to fetch or parse immediate parent post {parent_ref.uri}. Cannot determine parent author. Skipping reply.")
                        return # Skip if we can't verify parent
                except Exception as e:
                    logging.error(f"[Reply Check] Error fetching immediate parent post {parent_ref.uri}: {e}", exc_info=True)
                    return # Skip if fetch fails
            else:
                 logging.warning(f"[Reply Check] Notification {notification.uri} is a reply, but couldn't get parent ref from record. Skipping reply.")
                 return # Skip if structure is unexpected
        
        else: # Should not happen based on main loop filter, but good practice
            logging.warning(f"Skipping notification {notification.uri} with unexpected reason: {notification.reason}")
            return

        # --- Generate context and reply (if not returned/skipped above) ---
        context_string = format_thread_for_gemini(thread_view_of_mentioned_post, BLUESKY_HANDLE)
        if not context_string:
            logging.warning(f"Failed to generate context string for {mentioned_post_uri}. Skipping reply.")
            return

        dynamic_instruction = (
            "The following is a Bluesky conversation thread. "
            "Your primary task is to formulate a direct and relevant reply to the *final message* in this thread. "
            "Use the preceding messages only for context. "
            "Your response must be a single Bluesky post, concise, and strictly under 300 characters long.\\\\n\\\\n"
            "---BEGIN THREAD CONTEXT---\\\\n"
        )
        full_prompt_for_gemini = dynamic_instruction + context_string + "\\\\n---END THREAD CONTEXT---"
        
        logging.debug(f"Generated full prompt for Gemini:\\\\n{full_prompt_for_gemini}")
        
        gemini_response_obj = None # Renamed to avoid confusion with bsky_client.send_post response
        reply_text = None
        # Remove image-related variables as Flash doesn't generate images
        # image_data_bytes = None 
        # generated_alt_text = "Generated image" # Not used yet

        try:
            logging.info(f"Sending context to Gemini ({GEMINI_MODEL_NAME})...")
            gemini_response_obj = gemini_model_ref.generate_content(full_prompt_for_gemini)

            try:
                 reply_text = gemini_response_obj.text
            except ValueError as ve: 
                 logging.error(f"Gemini text generation failed for {mentioned_post_uri} (ValueError): {ve}")
                 if hasattr(gemini_response_obj, 'prompt_feedback') and gemini_response_obj.prompt_feedback.block_reason:
                     logging.error(f"Gemini prompt blocked. Reason: {gemini_response_obj.prompt_feedback.block_reason}")
                 # If text generation fails (e.g., blocked), we likely don't want to proceed.
                 return # Exit processing if text failed
            
            # Removed image checking logic here

        except Exception as e:
            logging.error(f"Gemini content generation failed for {mentioned_post_uri}: {e}", exc_info=True)
            return

        if not reply_text:
            logging.warning(f"Gemini returned no usable text content for {mentioned_post_uri}. Skipping reply.")
            return

        # Prepare post content (text, facets)
        post_text = reply_text # Ensure post_text is a string
        
        # Facet creation for mentions and links (simplified)
        facets = []
        if post_text: # Only create facets if there is text
            # Example: Detect @mentions (simple regex, can be improved)
            for match in re.finditer(r'@([a-zA-Z0-9_.-]+)', post_text):
                handle = match.group(1)
                # Ideally, resolve handle to DID here for robust facets
                # For simplicity, we'll assume it's a valid handle string for now.
                # Note: This is a naive implementation. For production, use DID resolution.
                facets.append(
                    at_models.AppBskyRichtextFacet.Main(
                        index=at_models.AppBskyRichtextFacet.ByteSlice(byteStart=match.start(), byteEnd=match.end()),
                        features=[at_models.AppBskyRichtextFacet.Mention(did=f"did:plc:{handle}")] # Placeholder DID!
                    )
                )
            # Example: Detect links (simple regex, can be improved)
            for match in re.finditer(r'(https?://[^\\s]+)', post_text):
                uri = match.group(1)
                facets.append(
                    at_models.AppBskyRichtextFacet.Main(
                        index=at_models.AppBskyRichtextFacet.ByteSlice(byteStart=match.start(), byteEnd=match.end()),
                        features=[at_models.AppBskyRichtextFacet.Link(uri=uri)]
                    )
                )
        
        # Removed image embedding logic
        embed_to_post = None

        # --- Determine Root and Parent for the Reply --- 
        # The parent of our reply is always the post that triggered the notification (target_post).
        parent_ref = at_models.ComAtprotoRepoStrongRef.Main(cid=target_post.cid, uri=target_post.uri)

        # Determine the root of the thread.
        root_ref = None
        post_record = target_post.record # Get the record data of the target post
        
        # Check if the target post's record is a standard post record and if it contains reply info
        if isinstance(post_record, at_models.AppBskyFeedPost.Record) and post_record.reply:
            # If target_post is itself a reply, use the root specified in its reply reference.
            root_ref = post_record.reply.root
            logging.debug(f"Target post {target_post.uri} is a reply. Using its root: {root_ref.uri}")
        
        # If target_post is not a reply, or if we couldn't get the root ref from its record,
        # then the target_post itself serves as the root for our reply.
        if root_ref is None:
            root_ref = parent_ref # Use the target_post's StrongRef as the root
            logging.debug(f"Target post {target_post.uri} is not a reply (or root ref missing). Using target post as root.")
        # --- End Determine Root and Parent --- 
        
        # Send the reply post
        logging.info(f"Sending reply to {mentioned_post_uri}: Text='{post_text[:50]}...', HasImage=False") # Set HasImage to False
        
        # Make sure facets is None if empty, not an empty list, as per SDK expectations for some versions.
        facets_to_send = facets if facets else None

        bsky_client.send_post(
            text=post_text,
            reply_to=at_models.AppBskyFeedPost.ReplyRef(root=root_ref, parent=parent_ref),
            embed=None, # Explicitly set embed to None
            facets=facets_to_send
        )
        logging.info(f"Successfully sent reply to {mentioned_post_uri}")

    except AtProtocolError as e:
        logging.error(f"Bluesky API error processing mention {mentioned_post_uri}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error processing mention {mentioned_post_uri}: {e}", exc_info=True)


def main_bot_loop():
    """Main loop for the bot to check for mentions and process them."""
    global bsky_client, gemini_model, processed_uris_this_run # Ensure access to initialized clients

    if not (bsky_client and gemini_model):
        logging.critical("Bluesky client or Gemini model not initialized. Exiting main loop.")
        return
    
    logging.info("Bot starting main loop...")
    while True:
        try:
            # Fetch notifications - no seenAt parameter, rely on isRead and updateSeen
            params = ListNotificationsParams(limit=NOTIFICATION_FETCH_LIMIT)
            response = bsky_client.app.bsky.notification.list_notifications(params=params)

            if response and response.notifications:
                latest_notification_indexed_at_in_batch = None # To track for updateSeen

                # Sort by indexedAt to process older unread notifications first,
                # and to correctly find the latest for updateSeen
                sorted_notifications = sorted(response.notifications, key=lambda n: n.indexed_at)

                for notification in sorted_notifications:
                    # Update latest_notification_indexed_at_in_batch with the timestamp of every notification seen in this batch
                    # This ensures updateSeen covers everything fetched, even if not processed (e.g. already read, or not a mention)
                    if notification.indexed_at:
                        if latest_notification_indexed_at_in_batch is None or notification.indexed_at > latest_notification_indexed_at_in_batch:
                            latest_notification_indexed_at_in_batch = notification.indexed_at
                    
                    if notification.is_read:
                        logging.debug(f"Skipping already read notification: {notification.uri}")
                        continue

                    # Skip if already processed in this run (handles is_read lag)
                    if notification.uri in processed_uris_this_run:
                        logging.debug(f"Skipping notification {notification.uri} already processed in this run.")
                        continue

                    # Skip if notification is from the bot itself to avoid loops or self-processing
                    if notification.author.handle == BLUESKY_HANDLE:
                        logging.info(f"Skipping notification {notification.uri} from bot ({BLUESKY_HANDLE}) itself.")
                        continue
                        
                    if notification.reason in ['mention', 'reply']:
                        # Pass the gemini_model directly
                        process_mention(notification, gemini_model) 
                    # else:
                        # logging.debug(f"Skipping notification {notification.uri}: Reason '{notification.reason}' not 'mention' or 'reply'.")

                # After processing all notifications in the batch, update seen status on server
                if latest_notification_indexed_at_in_batch:
                    try:
                        bsky_client.app.bsky.notification.update_seen({'seenAt': latest_notification_indexed_at_in_batch})
                        logging.info(f"Successfully called update_seen with server timestamp: {latest_notification_indexed_at_in_batch}")
                        # No longer saving timestamp to file
                    except Exception as e:
                        logging.error(f"Error calling update_seen: {e}", exc_info=True)
            else:
                logging.info("No new notifications found or error in fetching.")
        
        except AtProtocolError as e:
            logging.error(f"Bluesky API error in main loop: {e}")
            if "ExpiredToken" in str(e) or "InvalidToken" in str(e):
                logging.info("Attempting to re-login due to token error...")
                bsky_client = initialize_bluesky_client() # Re-initialize
                if not bsky_client:
                    logging.error("Failed to re-login. Waiting before retrying...")
                    time.sleep(60) # Wait longer if re-login fails
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}", exc_info=True)

        time.sleep(MENTION_CHECK_INTERVAL_SECONDS)

def main():
    global bsky_client, gemini_model # Declare intent to modify globals

    logging.info("Bot starting...")
    
    bsky_client = initialize_bluesky_client()
    gemini_model = initialize_gemini_model()

    # No initial update_seen call based on a loaded timestamp
    if bsky_client and gemini_model:
        main_bot_loop()
    else:
        if not bsky_client:
            logging.error("Failed to initialize Bluesky client. Bot cannot start.")
        if not gemini_model:
            logging.error("Failed to initialize Gemini model. Bot cannot start.")

if __name__ == "__main__":
    main()