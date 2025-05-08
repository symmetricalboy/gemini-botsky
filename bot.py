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
GEMINI_MODEL_NAME = "models/gemini-1.5-pro-latest" # Changed based on ListModels output

# Constants
BOT_SYSTEM_INSTRUCTION = "You are a helpful assistant on the Bluesky social network. Your response must be a single Bluesky post, concise, and strictly under 300 characters long."
MENTION_CHECK_INTERVAL_SECONDS = 15 # Check for new mentions every 15 seconds (was 60)
MAX_THREAD_DEPTH_FOR_CONTEXT = 15 # How many parent posts to fetch for context
NOTIFICATION_FETCH_LIMIT = 100 # How many notifications to fetch (was 25)

# Global variable to store the client, to be initialized in main()
bsky_client = None 
# Global variable to store the timestamp of the last processed mention
# Using a file for persistence across restarts would be more robust
last_processed_mention_ctime = None 

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

        # --- REMOVING TEMPORARY MODEL LISTING --- 
        # logging.info("--- Listing Available Gemini Models (for v1beta API) ---")
        # for m in genai.list_models():
        #    ...
        # logging.info("--- Finished Listing Available Models ---")
        # --- END REMOVAL --- 

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
        logging.info(f"Successfully initialized Gemini model with: {GEMINI_MODEL_NAME}") # Updated log message
        return model
    except Exception as e:
        logging.error(f"Failed to initialize Gemini model: {e}", exc_info=True) # Simplified error message
        return None

def format_thread_for_gemini(thread_view: models.AppBskyFeedDefs.ThreadViewPost, own_handle: str) -> str | None:
    """
    Formats the thread leading up to and including the mentioned_post into a string for Gemini.
    `thread_view` is the ThreadViewPost for the post that contains the mention.
    """
    history = []
    current_view = thread_view

    # Traverse up the parent chain from the mentioned post
    while current_view:
        if isinstance(current_view, models.AppBskyFeedDefs.ThreadViewPost) and current_view.post:
            post_record = current_view.post.record
            # We are interested in app.bsky.feed.post records with text
            if isinstance(post_record, models.AppBskyFeedPost.Record) and hasattr(post_record, 'text'):
                author_display_name = current_view.post.author.display_name or current_view.post.author.handle
                text = post_record.text
                # Optional: filter out bot's own previous messages if own_handle == current_view.post.author.handle
                history.append(f"{author_display_name} (@{current_view.post.author.handle}): {text}")
        elif isinstance(current_view, (models.AppBskyFeedDefs.NotFoundPost, models.AppBskyFeedDefs.BlockedPost)):
            logging.warning(f"Encountered NotFoundPost or BlockedPost while traversing thread parent: {current_view}")
            break # Stop if we hit a part of the thread that's not accessible
        
        # Move to the parent
        if hasattr(current_view, 'parent') and current_view.parent:
            current_view = current_view.parent
        else:
            break # No more parents

    history.reverse() # Order from root to the mentioned post
    
    if not history:
        logging.warning("Could not construct any context from the thread.")
        # Attempt to use the main post text directly if it's a text post
        if isinstance(thread_view.post.record, models.AppBskyFeedPost.Record) and hasattr(thread_view.post.record, 'text'):
            author_display_name = thread_view.post.author.display_name or thread_view.post.author.handle
            return f"{author_display_name} (@{thread_view.post.author.handle}): {thread_view.post.record.text}"
        return None # "Unable to retrieve thread context." -> handled by None return
        
    return "\n\n".join(history)

def process_mention(notification: at_models.AppBskyNotificationListNotifications.Notification, gemini_model: genai.GenerativeModel):
    """Processes a single mention notification."""
    global bsky_client
    if not bsky_client:
        logging.error("Bluesky client not initialized. Cannot process mention.")
        return

    mentioned_post_uri = notification.uri
    logging.info(f"Processing mention/reply in post: {mentioned_post_uri}")

    try:
        # Fetch the thread containing the mention/reply
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
            
        # --- Start: Checks for skipping (Direct Reply / Duplicate) ---
        already_replied = False 
        if notification.reason == 'reply':
            parent_view = thread_view_of_mentioned_post.parent
            if isinstance(parent_view, at_models.AppBskyFeedDefs.ThreadViewPost) and parent_view.post:
                parent_post = parent_view.post
                if parent_post.author.handle != BLUESKY_HANDLE:
                    logging.info(f"Skipping reply notification {notification.uri}: Parent post {parent_post.uri} not authored by bot ({BLUESKY_HANDLE}).")
                    return 
                
                # Parent is by bot, check for duplicates
                logging.debug(f"Reply {notification.uri} is to a bot post {parent_post.uri}. Checking for duplicates.")
                parent_uri = parent_post.uri
                try:
                    parent_params = GetPostThreadParams(uri=parent_uri, depth=1)
                    parent_thread_response = bsky_client.app.bsky.feed.get_post_thread(params=parent_params)
                    if isinstance(parent_thread_response.thread, at_models.AppBskyFeedDefs.ThreadViewPost) and parent_thread_response.thread.replies:
                        for reply_to_parent in parent_thread_response.thread.replies:
                            if reply_to_parent.post and reply_to_parent.post.author and reply_to_parent.post.author.handle == BLUESKY_HANDLE:
                                already_replied = True
                                logging.info(f"Detected existing reply by bot ({BLUESKY_HANDLE}) to parent post {parent_uri}. Skipping reply to {target_post.uri}.")
                                break 
                    else:
                        logging.debug(f"Parent post {parent_uri} has no replies according to its thread view.")
                except AtProtocolError as e:
                    logging.error(f"Failed to fetch parent thread {parent_uri} to check for duplicates: {e}")
                    already_replied = True 
                    logging.warning(f"Skipping reply to {target_post.uri} due to error checking parent.")
                except Exception as e:
                    logging.error(f"Unexpected error fetching parent thread {parent_uri}: {e}", exc_info=True)
                    already_replied = True 
                    logging.warning(f"Skipping reply to {target_post.uri} due to error checking parent.")
            else:
                logging.warning(f"Could not identify parent post for reply notification {notification.uri}. Skipping reply for safety." )
                already_replied = True 

        elif notification.reason == 'mention':
            if thread_view_of_mentioned_post.replies:
                for reply in thread_view_of_mentioned_post.replies:
                    if reply.post and reply.post.author and reply.post.author.handle == BLUESKY_HANDLE:
                        already_replied = True
                        logging.info(f"Detected existing reply by bot to mentioned post {target_post.uri}. Skipping duplicate reply.")
                        break
                        
        if already_replied:
            return 
        # --- End: Checks for skipping ---

        # Construct context for Gemini
        context_string = format_thread_for_gemini(thread_view_of_mentioned_post, BLUESKY_HANDLE)
        if not context_string:
            logging.warning(f"Failed to generate context string for {mentioned_post_uri}. Skipping reply.")
            return

        # NEW: Construct the full prompt for Gemini
        # The BOT_SYSTEM_INSTRUCTION is set at model initialization.
        # This dynamic instruction guides the model for *this specific request*.
        dynamic_instruction = (
            "The following is a Bluesky conversation thread. "
            "Your primary task is to formulate a direct and relevant reply to the *final message* in this thread. "
            "Use the preceding messages only for context. "
            "Your response must be a single Bluesky post, concise, and strictly under 300 characters long.\n\n"
            "---BEGIN THREAD CONTEXT---\n"
        )
        # Ensure there's a clear separation if context_string might also have delimiters.
        # For now, simple concatenation is fine.
        full_prompt_for_gemini = dynamic_instruction + context_string + "\n---END THREAD CONTEXT---"
        
        logging.debug(f"Generated full prompt for Gemini:\n{full_prompt_for_gemini}")
        
        # Get response from Gemini (potentially image + text)
        gemini_response = None
        reply_text = None
        image_data_bytes = None
        generated_alt_text = "Generated image" # Default alt text

        try:
            logging.info(f"Sending context to Gemini ({GEMINI_MODEL_NAME})...")
            # Assume generate_content handles multimodal prompts correctly
            gemini_response = gemini_model.generate_content(full_prompt_for_gemini) # Use the new full_prompt_for_gemini

            # Extract text part first
            try:
            reply_text = gemini_response.text
            except ValueError as ve: 
                 logging.error(f"Gemini text generation failed for {mentioned_post_uri} (ValueError): {ve}")
            if hasattr(gemini_response, 'prompt_feedback') and gemini_response.prompt_feedback.block_reason:
                logging.error(f"Gemini prompt blocked. Reason: {gemini_response.prompt_feedback.block_reason}")
                 # Proceed only if we might get an image
                 reply_text = "" # Allow empty text if image might exist
            
            # Extract image data if present
            if hasattr(gemini_response, 'parts'):
                for part in gemini_response.parts:
                    # Check for inline image data
                    if hasattr(part, 'inline_data') and part.inline_data and hasattr(part.inline_data, 'mime_type') and part.inline_data.mime_type.startswith('image/'):
                        if hasattr(part.inline_data, 'data') and isinstance(part.inline_data.data, bytes):
                             image_data_bytes = part.inline_data.data
                             logging.info(f"Received image data ({part.inline_data.mime_type}, {len(image_data_bytes)} bytes) from Gemini.")
                             # TODO: Potentially extract alt text if Gemini provides it in another part or field
                             break # Assume only one image for now
                        else:
                             logging.warning("Found image part in Gemini response, but 'data' attribute missing or not bytes.")
                    # TODO: Potentially handle other image response types (e.g., function calls, URLs) if needed

        except Exception as e:
            logging.error(f"Gemini API call failed for {mentioned_post_uri}: {e}", exc_info=True)
            return # Exit if Gemini call itself fails

        # Check if we got *anything* back
        if not reply_text and not image_data_bytes:
             logging.info(f"Gemini returned no usable text or image for {mentioned_post_uri}. Skipping reply.")
            return

        logging.info(f'Gemini reply text for {mentioned_post_uri}: "{reply_text[:50]}..."')
        if image_data_bytes:
             logging.info(f'Gemini also provided image data.') # Don't log bytes here

        # --- Embed Generation (if image exists) ---
        embed_to_send = None
        if image_data_bytes:
             try:
                 logging.info("Uploading image blob to Bluesky...")
                 upload_response = bsky_client.upload_blob(image_data_bytes)
                 
                 if upload_response and upload_response.blob:
                      logging.info(f"Image blob uploaded successfully: CID={upload_response.blob.ref.cid}") # Access cid via ref
                      
                      image_to_embed = at_models.AppBskyEmbedImages.Image(
                           alt=generated_alt_text, 
                           image=upload_response.blob # Pass the blob object directly
                      )
                      embed_to_send = at_models.AppBskyEmbedImages.Main(images=[image_to_embed])
                      logging.info("Created image embed structure.")
                 else:
                      logging.error("Failed to upload image blob or blob reference missing in response.")

             except AtProtocolError as e:
                 logging.error(f"Bluesky API error during image blob upload: {e}")
             except Exception as e:
                 logging.error(f"Unexpected error during image blob upload or embed creation: {e}", exc_info=True)
                 # Continue without embed if upload failed

        # --- Facet Generation Start ---
        facets = []
        try:
            # --- Step 1: Mentions ---
            # Regex to find handles (including the leading @)
            handle_regex = r'@((?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\\.){1,}[a-zA-Z]{2,})\''
            reply_text_bytes = reply_text.encode('utf-8')
            used_byte_ranges = [] # Keep track of used ranges to prevent overlaps

            for match in re.finditer(handle_regex, reply_text):
                handle_with_at = match.group(0)
                handle_only = match.group(1)
                
                # Calculate byte indices based on byte string
                mention_bytes = handle_with_at.encode('utf-8')
                byte_start = reply_text_bytes.find(mention_bytes, match.start())
                if byte_start != -1:
                    byte_end = byte_start + len(mention_bytes)
                else:
                    logging.warning(f"Could not find byte offset for mention '{handle_with_at}' in reply text. Skipping facet.")
                    continue
                
                # Check for overlap - skip if this range is already covered
                if any(max(byte_start, r[0]) < min(byte_end, r[1]) for r in used_byte_ranges):
                    logging.debug(f"Skipping mention facet for {handle_with_at} due to overlap.")
                    continue

                logging.info(f"Found potential mention: {handle_with_at} (bytes {byte_start}-{byte_end})")

                try:
                    resolve_response = bsky_client.resolve_handle(handle=handle_only)
                    resolved_did = resolve_response.did
                    logging.info(f"Resolved {handle_only} to DID: {resolved_did}")
                    
                    mention_feature = at_models.AppBskyRichtextFacet.Mention(did=resolved_did)
                    facet = at_models.AppBskyRichtextFacet.Main(
                        index=at_models.AppBskyRichtextFacet.ByteSlice(byte_start=byte_start, byte_end=byte_end),
                        features=[mention_feature]
                    )
                    facets.append(facet)
                    used_byte_ranges.append((byte_start, byte_end)) # Mark range as used
                except AtProtocolError as e:
                    logging.warning(f"Failed to resolve handle '{handle_only}': {e}. Mention will be plain text.")
                except Exception as e:
                    logging.error(f"Unexpected error resolving handle '{handle_only}': {e}", exc_info=True)

            # --- Step 2: Links ---
            # Simple regex for http/https URLs
            # Note: More complex regex needed for broader URL patterns, but keep it simple for now.
            link_regex = r'https?://[\\S]+' 
            for match in re.finditer(link_regex, reply_text):
                url = match.group(0)
                
                # Calculate byte indices
                link_bytes = url.encode('utf-8')
                byte_start = reply_text_bytes.find(link_bytes, match.start())
                if byte_start != -1:
                    byte_end = byte_start + len(link_bytes)
                else:
                    logging.warning(f"Could not find byte offset for link '{url}'. Skipping facet.")
                    continue

                # Check for overlap - skip if this range is already covered
                if any(max(byte_start, r[0]) < min(byte_end, r[1]) for r in used_byte_ranges):
                    logging.debug(f"Skipping link facet for {url} due to overlap.")
                    continue
                
                logging.info(f"Found potential link: {url} (bytes {byte_start}-{byte_end})")
                try:
                    # Basic validation (already done by regex mostly, but good practice)
                    if not url.startswith(('http://', 'https://')): 
                         logging.warning(f"Skipping link facet for {url}: Does not start with http(s)://")
                         continue

                    link_feature = at_models.AppBskyRichtextFacet.Link(uri=url)
                    facet = at_models.AppBskyRichtextFacet.Main(
                         index=at_models.AppBskyRichtextFacet.ByteSlice(byte_start=byte_start, byte_end=byte_end),
                         features=[link_feature]
                    )
                    facets.append(facet)
                    used_byte_ranges.append((byte_start, byte_end)) # Mark range as used
                except Exception as e:
                     logging.error(f"Unexpected error creating link facet for {url}: {e}", exc_info=True)


            # --- Step 3: Hashtags ---
            # Regex for hashtags (simple version: # followed by word characters)
            tag_regex = r'#([\\w]+)'
            for match in re.finditer(tag_regex, reply_text):
                tag_with_hash = match.group(0)
                tag_only = match.group(1)
                
                # Calculate byte indices
                tag_bytes = tag_with_hash.encode('utf-8')
                byte_start = reply_text_bytes.find(tag_bytes, match.start())
                if byte_start != -1:
                     byte_end = byte_start + len(tag_bytes)
                else:
                     logging.warning(f"Could not find byte offset for tag '{tag_with_hash}'. Skipping facet.")
                     continue

                # Check for overlap - skip if this range is already covered
                if any(max(byte_start, r[0]) < min(byte_end, r[1]) for r in used_byte_ranges):
                    logging.debug(f"Skipping tag facet for {tag_with_hash} due to overlap.")
                    continue

                logging.info(f"Found potential hashtag: {tag_with_hash} (bytes {byte_start}-{byte_end})")
                try:
                    tag_feature = at_models.AppBskyRichtextFacet.Tag(tag=tag_only)
                    facet = at_models.AppBskyRichtextFacet.Main(
                         index=at_models.AppBskyRichtextFacet.ByteSlice(byte_start=byte_start, byte_end=byte_end),
                         features=[tag_feature]
                    )
                    facets.append(facet)
                    used_byte_ranges.append((byte_start, byte_end)) # Mark range as used
                except Exception as e:
                    logging.error(f"Unexpected error creating tag facet for {tag_with_hash}: {e}", exc_info=True)


        except Exception as e:
            logging.error(f"Error during facet generation for {mentioned_post_uri}: {e}", exc_info=True)
        # --- Facet Generation End ---

        # Determine root and parent for the reply
        parent_strong_ref = at_models.ComAtprotoRepoStrongRef.Main(uri=target_post.uri, cid=target_post.cid)
        if target_post.record and isinstance(target_post.record, at_models.AppBskyFeedPost.Record) and target_post.record.reply:
            root_ref_input = target_post.record.reply.root
        else: 
            root_ref_input = target_post
        root_strong_ref = at_models.ComAtprotoRepoStrongRef.Main(uri=root_ref_input.uri, cid=root_ref_input.cid)
        reply_ref = at_models.AppBskyFeedPost.ReplyRef(root=root_strong_ref, parent=parent_strong_ref)

        # Post the reply, including facets AND embed
        try:
            logging.info(f"Sending post in reply to {target_post.uri}...")
            bsky_client.send_post(
                text=reply_text,
                reply_to=reply_ref,
                facets=facets if facets else None,
                embed=embed_to_send # Add the embed here
            )
        logging.info(f"Successfully posted reply to {mentioned_post_uri}")
        except AtProtocolError as e:
             logging.error(f"Bluesky API error sending post reply to {mentioned_post_uri}: {e}")
        except Exception as e:
             logging.error(f"Unexpected error sending post reply to {mentioned_post_uri}: {e}", exc_info=True)

    except AtProtocolError as e:
        logging.error(f"Bluesky API error during processing of {mentioned_post_uri}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while processing mention/reply {mentioned_post_uri}: {e}", exc_info=True)


def main_bot_loop():
    """Main loop for the bot to check for mentions and reply."""
    global bsky_client, last_processed_mention_ctime
    
    gemini_model = initialize_gemini_model()
    if not gemini_model:
        return # Exit if Gemini can't be initialized

    if not bsky_client: # Should be initialized by main() before calling this
        logging.error("Bluesky client not available in main loop.")
        return

    logging.info("Bot starting main loop...")
    
    # Initialize last_processed_mention_ctime.
    # For a more robust solution, this could be read from a file.
    # For now, it starts as None, meaning all unread mentions up to limit will be processed on first run.
    # After first run, it will be the timestamp of the latest processed mention.

    while True:
        try:
            # Fetch notifications
            # The `seenAt` parameter in list_notifications is for marking them as read,
            # not for filtering. We need to filter by notification.indexedAt (ctime) ourselves.
            # notifications_response = bsky_client.app.bsky.notification.list_notifications(limit=25) # Fetch recent 25
            
            # Use explicit Params object with updated limit
            params = ListNotificationsParams(limit=NOTIFICATION_FETCH_LIMIT)
            notifications_response = bsky_client.app.bsky.notification.list_notifications(params=params)
            
            # <<< START NEW DEBUG LOGGING >>>
            logging.debug(f"Type of notifications_response: {type(notifications_response)}")
            if notifications_response:
                logging.debug(f"Notifications response object: {notifications_response}") # Log the whole object (might be verbose)
                logging.debug(f"Does response have 'notifications' attribute? {hasattr(notifications_response, 'notifications')}")
                if hasattr(notifications_response, 'notifications'):
                    logging.debug(f"Value of notifications attribute: {notifications_response.notifications}")
                    logging.debug(f"Type of notifications attribute: {type(notifications_response.notifications)}")
                    logging.debug(f"Is notifications attribute None? {notifications_response.notifications is None}")
                    if isinstance(notifications_response.notifications, list):
                         logging.debug(f"Length of notifications list: {len(notifications_response.notifications)}")
            else:
                 logging.debug("Notifications response object is None or evaluates to False.")
            # <<< END NEW DEBUG LOGGING >>>
            
            if not notifications_response or not notifications_response.notifications:
                logging.debug("No new notifications found (or response/notifications attribute was empty/None).") # Updated log message
                time.sleep(MENTION_CHECK_INTERVAL_SECONDS)
                continue

            # new_mentions_to_process = [] # Old list
            mentions_to_process_with_ts = [] # New list to store (timestamp, mention) tuples
            
            # latest_ctime_in_batch = last_processed_mention_ctime # This wasn't used

            logging.debug(f"Fetched {len(notifications_response.notifications)} notifications. Checking against last processed ctime: {last_processed_mention_ctime}")

            for notification in notifications_response.notifications:
                # Attempt to get indexedAt gracefully
                indexed_at_value = None
                raw_notification_data = {} # For logging if needed
                try:
                    # Try direct attribute access first (expected)
                    if hasattr(notification, 'indexedAt') and isinstance(notification.indexedAt, str):
                        indexed_at_value = notification.indexedAt
                    else:
                        # Fallback: Check underlying dict representation (common in pydantic models)
                        raw_notification_data = notification.model_dump() # Use model_dump
                        if isinstance(raw_notification_data.get('indexedAt'), str):
                            indexed_at_value = raw_notification_data.get('indexedAt')
                        elif isinstance(raw_notification_data.get('indexed_at'), str): # Check snake_case
                             indexed_at_value = raw_notification_data.get('indexed_at')
                        
                except Exception as e:
                     logging.warning(f"Error accessing notification data for {getattr(notification, 'uri', '[URI UNKNOWN]')}: {e}")

                logging.debug(f"Checking notification: URI={getattr(notification, 'uri', 'N/A')}, Reason={getattr(notification, 'reason', 'N/A')}, Author={getattr(notification.author, 'handle', 'N/A')}, Found IndexedAt='{indexed_at_value}'")

                # Check 1: Valid indexedAt found?
                if not indexed_at_value:
                    logging.debug(f" -> Skipping notification {getattr(notification, 'cid', 'N/A')}: Could not find valid 'indexedAt' string field.")
                    if raw_notification_data: # Log raw data if we tried accessing dict
                         logging.debug(f"   Raw notification data: {raw_notification_data}")
                    elif hasattr(notification, 'dict'):
                         logging.debug(f"   Raw notification data (dict): {notification.dict()}")
                    elif hasattr(notification, 'model_dump'):
                         logging.debug(f"   Raw notification data (model_dump): {notification.model_dump()}")
                    continue

                # Check 2: Is it a mention OR a reply?
                if notification.reason not in ['mention', 'reply']:
                    logging.debug(f" -> Skipping notification {notification.uri}: reason is not 'mention' or 'reply' ({notification.reason}).")
                            continue

                # Check 3: Is it newer than the last processed one?
                current_indexed_at = indexed_at_value # Use the found value
                is_new = last_processed_mention_ctime is None or current_indexed_at > last_processed_mention_ctime
                logging.debug(f" -> Checking if new: Current IndexedAt={current_indexed_at}, Last Processed={last_processed_mention_ctime}, Is New={is_new}")
                if not is_new:
                    logging.debug(f" -> Skipping mention {notification.uri}: not newer than last processed.")
                    continue

                # Check 4: Is it a mention/reply *by* the bot itself?
                if notification.author.handle == BLUESKY_HANDLE:
                    logging.debug(f" -> Skipping notification {notification.uri}: notification author is the bot itself.")
                    continue

                # If all checks passed:
                logging.debug(f" -> Adding new {notification.reason} from {notification.author.handle} ({current_indexed_at}) to process list: {notification.uri}")
                # new_mentions_to_process.append(notification) # Old way
                mentions_to_process_with_ts.append((current_indexed_at, notification)) # Store as tuple

            # Process the found mentions (oldest first based on sort)
            # new_mentions_to_process.sort(key=lambda n: n.indexedAt) # Old sort - caused error
            mentions_to_process_with_ts.sort(key=lambda item: item[0]) # Sort by timestamp

            if not mentions_to_process_with_ts:
                 logging.debug(f"No mentions found in this batch meeting criteria (new and not self-mention). Last processed ctime: {last_processed_mention_ctime}")

            # for mention in new_mentions_to_process: # Old iteration
            for timestamp, mention in mentions_to_process_with_ts: # Iterate through sorted tuples
                # Add logging at the start of process_mention as well
                logging.info(f"Starting processing for mention: {mention.uri} (Timestamp: {timestamp})") 
                process_mention(mention, gemini_model)
                # Update last processed time
                # Use the timestamp we already have from the tuple
                if last_processed_mention_ctime is None or timestamp > last_processed_mention_ctime:
                    last_processed_mention_ctime = timestamp
            
            if mentions_to_process_with_ts:
                logging.info(f"Finished processing {len(mentions_to_process_with_ts)} new mentions. Last processed ctime updated to: {last_processed_mention_ctime}")
            # else:
                # logging.debug(f"No new, unread mentions found. Current last processed ctime: {last_processed_mention_ctime}") # Removed redundant log

        except AtProtocolError as e:
            # Attempt to access e.response and e.response.status_code carefully
            status_code = None
            if hasattr(e, 'response') and e.response and hasattr(e.response, 'status_code'):
                status_code = e.response.status_code
            
            if status_code == 401: # Unauthorized
                 logging.error(f"Bluesky authentication error: {e}. Attempting to re-login...")
                 bsky_client = initialize_bluesky_client()
                 if not bsky_client:
                     logging.error("Re-login failed. Exiting.")
                     break # Exit loop if re-login fails
            else:
                logging.error(f"Bluesky API error in main loop: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred in the main bot loop: {e}", exc_info=True)
        
        logging.debug(f"Sleeping for {MENTION_CHECK_INTERVAL_SECONDS} seconds...")
        time.sleep(MENTION_CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    logging.info("Bot starting...")
    if not all([BLUESKY_HANDLE, BLUESKY_PASSWORD, GEMINI_API_KEY]):
        logging.error("Missing one or more critical environment variables: BLUESKY_HANDLE, BLUESKY_PASSWORD, GEMINI_API_KEY. Exiting.")
    else:
        bsky_client = initialize_bluesky_client()
        if bsky_client:
            main_bot_loop()
        else:
            logging.error("Failed to initialize Bluesky client. Bot cannot start.")
    logging.info("Bot shutting down.")