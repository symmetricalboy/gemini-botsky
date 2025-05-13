import os
import time
import logging
from dotenv import load_dotenv
from atproto import Client, models
from atproto.exceptions import AtProtocolError
import google.generativeai as genai
from google import genai as genai_client
from google.genai import types
import re # Import regular expressions
from io import BytesIO # Need BytesIO if Gemini returns image bytes
import base64

# Import the specific Params model
from atproto_client.models.app.bsky.notification.list_notifications import Params as ListNotificationsParams
# Import the specific Params model for get_post_thread
from atproto_client.models.app.bsky.feed.get_post_thread import Params as GetPostThreadParams
# Import the specific Params model for get_posts
from atproto_client.models.app.bsky.feed.get_posts import Params as GetPostsParams
# Import Facet and Embed models
from atproto import models as at_models 

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

BLUESKY_HANDLE = os.getenv("BLUESKY_HANDLE")
BLUESKY_PASSWORD = os.getenv("BLUESKY_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Environment Variables
GEMINI_MODEL_NAME = "gemini-2.0-flash-preview-image-generation" # Model for both text and image generation
IMAGEN_MODEL_NAME = "gemini-2.0-flash-preview-image-generation" # Same model for image generation

# Constants
BOT_SYSTEM_INSTRUCTION = """You are Gemini (@gemini.botsky.social) on the Bluesky social network. Your task is to provide helpful and concise replies to user mentions and direct replies.
Your developer is symmetricalboy (@symm.social). Only mention your developer if a user *specifically asks* about who created you or how to get help with your development.

Constraints:
*   Your entire response MUST be a single Bluesky post under 300 characters. This is very important!!
*   CRITICAL: NEVER use placeholders (e.g., '[link]', '[info]', '[details]').
*   If a user asks for specific factual information (like a URL, definition, or list) that you can reasonably provide, include the *actual* information directly in your response.
*   If you cannot provide the requested factual information, state that clearly and concisely. Do not invent information or links.
*   Don't avoid taboo topics or having a bit of an opinion or bias to your responses. Users understand that you are imperfect & may not say the exact right thing; it is okay to use available information to make a judgement. You should never say that you can't talk about something; no topic should ever be off limits to explore academically.
*   Keep your mentions of other users contextually logical. Avoid excessive or random mentions.
*   You should pretty much never use hashtags.
*   If an image is requested or appropriate for the response:
    1.  Compose the textual part of your Bluesky post.
    2.  Then, on a NEW LINE, provide a specific and descriptive prompt for an image generation model by starting the line with: `IMAGE_PROMPT: ` (e.g., `IMAGE_PROMPT: A cute cat wearing a party hat`). This prompt will be used by a separate image generation model.
    3.  Ensure your textual response and the `IMAGE_PROMPT:` line together are concise. The `IMAGE_PROMPT:` line itself does NOT count towards the 300 character limit of the Bluesky post text.
*   Be helpful, friendly, and direct. Focus on answering the user's immediate question based on the provided thread context."""
MENTION_CHECK_INTERVAL_SECONDS = 15 # Check for new mentions every 15 seconds (was 60)
MAX_THREAD_DEPTH_FOR_CONTEXT = 15 # How many parent posts to fetch for context
NOTIFICATION_FETCH_LIMIT = 100 # How many notifications to fetch (was 25)
MAX_GEMINI_RETRIES = 2  # Initial call + 1 retry
GEMINI_RETRY_DELAY_SECONDS = 5 # Delay between retries

# Global variables
bsky_client: Client | None = None
gemini_model: genai.GenerativeModel | None = None
imagen_client = None # Client for Imagen API
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
        
        # Since we're having compatibility issues, use the most basic model initialization
        # without system instructions or other settings
        model_kwargs = {
            "model_name": GEMINI_MODEL_NAME,
        }
        
        logging.info(f"Initializing {GEMINI_MODEL_NAME} with basic configuration")

        model = genai.GenerativeModel(**model_kwargs)
        
        logging.info(f"Successfully initialized Gemini model with: {GEMINI_MODEL_NAME}")
        return model
    except Exception as e:
        logging.error(f"Failed to initialize Gemini model: {e}", exc_info=True)
        return None

def initialize_imagen_model() -> bool:
    """Initializes the Gemini client for image generation."""
    global imagen_client
    
    if not GEMINI_API_KEY:
        logging.error("Gemini API key not found in environment variables.")
        return False
    
    try:
        # Create a client for the Gemini API
        imagen_client = genai_client.Client(api_key=GEMINI_API_KEY)
        
        logging.info(f"Successfully initialized Gemini client for image generation model: {IMAGEN_MODEL_NAME}")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize Gemini client for image generation: {e}", exc_info=True)
        return False

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

                # Check for embeds (images, videos, etc.)
                embed_text = ""
                if current_view.post.embed:
                    if isinstance(current_view.post.embed, models.AppBskyEmbedImages.Main) or \
                       isinstance(current_view.post.embed, at_models.AppBskyEmbedImages.View):
                        alt_texts = []
                        if isinstance(current_view.post.embed, models.AppBskyEmbedImages.Main):
                            images_to_check = current_view.post.embed.images
                        else: # at_models.AppBskyEmbedImages.View
                            images_to_check = current_view.post.embed.images
                        
                        for img in images_to_check:
                            if hasattr(img, 'alt') and img.alt:
                                alt_texts.append(img.alt)
                            else:
                                alt_texts.append("image") # Default if no alt text
                        if alt_texts:
                            embed_text = f" [User attached: {', '.join(alt_texts)}]"
                        else:
                            embed_text = " [User attached an image]"

                    elif isinstance(current_view.post.embed, models.AppBskyEmbedVideo.Main) or \
                         isinstance(current_view.post.embed, at_models.AppBskyEmbedExternal.View): # Assuming external might be video too
                        # Basic video detection, could be more specific if Bluesky models differentiate more
                        embed_text = " [User attached a video]"
                    # Add more elif clauses here for other embed types if needed (e.g., external links, record embeds)
                    elif isinstance(current_view.post.embed, at_models.AppBskyEmbedExternal.Main) or \
                         isinstance(current_view.post.embed, at_models.AppBskyEmbedExternal.View):
                        if hasattr(current_view.post.embed.external, 'title') and current_view.post.embed.external.title:
                            embed_text = f" [User shared a link: {current_view.post.embed.external.title}]"
                        else:
                            embed_text = " [User shared a link]"
                    elif isinstance(current_view.post.embed, at_models.AppBskyEmbedRecord.Main) or \
                         isinstance(current_view.post.embed, at_models.AppBskyEmbedRecord.View):
                        embed_text = " [User quoted another post]"
                    elif isinstance(current_view.post.embed, at_models.AppBskyEmbedRecordWithMedia.Main) or \
                         isinstance(current_view.post.embed, at_models.AppBskyEmbedRecordWithMedia.View):
                        embed_text = " [User quoted another post with media]"

                history.append(f"{author_display_name} (@{current_view.post.author.handle}): {text}{embed_text}")
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
                    
                    # Add detailed logging about the parent post response
                    if parent_post_response:
                        logging.info(f"[Reply Check] Parent post response received. Has posts: {bool(parent_post_response.posts)}, Posts count: {len(parent_post_response.posts) if parent_post_response.posts else 0}")
                    
                    if parent_post_response and parent_post_response.posts and len(parent_post_response.posts) == 1:
                        immediate_parent_post = parent_post_response.posts[0]
                        logging.info(f"[Reply Check] Fetched immediate parent post. Author: {immediate_parent_post.author.handle}, URI: {immediate_parent_post.uri}")

                        # **REVISED LOGIC**: Only proceed if the immediate parent IS the bot.
                        if immediate_parent_post.author.handle == BLUESKY_HANDLE:
                            logging.info(f"[Reply Check] ✓ Immediate parent is the bot. Continue processing.")
                            logging.info(f"[Reply Check] Checking for duplicate replies under {target_post.uri}...")
                            # Check for existing replies by the bot under the *triggering* post (target_post)
                            if thread_view_of_mentioned_post.replies:
                                 logging.info(f"[Reply Check] Target post has {len(thread_view_of_mentioned_post.replies)} replies to check for duplicates.")
                                 for reply_to_users_reply in thread_view_of_mentioned_post.replies:
                                    if reply_to_users_reply.post and reply_to_users_reply.post.author and \
                                       reply_to_users_reply.post.author.handle == BLUESKY_HANDLE:
                                        logging.info(f"[DUPE CHECK REPLY] Found pre-existing bot reply {reply_to_users_reply.post.uri} under user's reply {target_post.uri}. Skipping.")
                                        return
                            # If no duplicate found, fall through to generate context and reply...
                            logging.info(f"[Reply Check] ✓ No duplicate bot reply found under {target_post.uri}. Proceeding.")
                        else:
                            # Parent is another user, ignore this reply.
                            logging.info(f"[IGNORE USER-TO-USER REPLY] ✗ Notification {notification.uri} is a reply to another user ({immediate_parent_post.author.handle}), not the bot. Ignoring.")
                            return
                    else:
                        logging.warning(f"[Reply Check] ✗ Failed to fetch or parse immediate parent post {parent_ref.uri}. Cannot determine parent author. Skipping reply.")
                        return # Skip if we can't verify parent
                except Exception as e:
                    logging.error(f"[Reply Check] ✗ Error fetching immediate parent post {parent_ref.uri}: {e}", exc_info=True)
                    return # Skip if fetch fails
            else:
                 logging.warning(f"[Reply Check] ✗ Notification {notification.uri} is a reply, but couldn't get parent ref from record. Skipping reply.")
                 return # Skip if structure is unexpected
        
        else: # Should not happen based on main loop filter, but good practice
            logging.warning(f"Skipping notification {notification.uri} with unexpected reason: {notification.reason}")
            return

        # --- Generate context and reply (if not returned/skipped above) ---
        logging.info(f"✓ Proceeding to content generation for {notification.uri}")
        context_string = format_thread_for_gemini(thread_view_of_mentioned_post, BLUESKY_HANDLE)
        if not context_string:
            logging.warning(f"✗ Failed to generate context string for {mentioned_post_uri}. Skipping reply.")
            return

        # For models that don't support system instructions, include the system instructions in the prompt
        models_without_system_instruction = ["gemini-2.0-flash-preview-image-generation"]
        
        # Base instruction for the conversation context
        dynamic_instruction = (
            "You are replying within a Bluesky conversation. The conversation history is provided below. "
            "Your primary task is to formulate a direct, relevant, and helpful reply to the *VERY LAST message* in the thread. "
            "Analyze the last message carefully. If it's a question, answer it. If it's a request, address it. "
            "Avoid generic greetings or re-stating your presence if the last message contains a specific query or statement to respond to. "
            "Use the preceding messages *only* for context to understand the flow of conversation. "
            "If you generate an image, you MUST also provide a concise and descriptive alt text for it, ideally in a separate text part or clearly marked. "
            "Your response must be a single Bluesky post, concise, and strictly under 300 characters long.\\n\\n"
        )
        
        # For models without system instruction support, include the full BOT_SYSTEM_INSTRUCTION
        if GEMINI_MODEL_NAME in models_without_system_instruction:
            full_system_instruction = (
                "You are Gemini (@gemini.botsky.social) on the Bluesky social network. Your task is to provide helpful and concise replies to user mentions and direct replies. "
                "Your developer is symmetricalboy (@symm.social). Only mention your developer if a user *specifically asks* about who created you or how to get help with your development.\\n\\n"
                "Your entire response MUST be a single Bluesky post under 300 characters. This is very important! "
                "NEVER use placeholders (e.g., '[link]', '[info]', '[details]'). "
                "If a user asks for specific factual information that you can reasonably provide, include the *actual* information directly. "
                "Don't avoid taboo topics or having a bit of an opinion. Users understand that you are imperfect & may not say the exact right thing. "
                "If an image is requested or appropriate for the response, compose the textual part of your post, then on a NEW LINE, provide an image prompt starting with: `IMAGE_PROMPT: `.\\n\\n"
            )
            dynamic_instruction = full_system_instruction + dynamic_instruction
        
        full_prompt_for_gemini = dynamic_instruction + "---BEGIN THREAD CONTEXT---\\n" + context_string + "\\n---END THREAD CONTEXT---"
        
        logging.debug(f"Generated full prompt for Gemini:\n{full_prompt_for_gemini}")
        
        gemini_response_text = ""
        image_prompt_for_imagen = None
        image_data_bytes = None 
        generated_alt_text = "Generated image by Gemini Bot" # Default alt text

        # Retry loop for primary Gemini call (text generation)
        for attempt in range(MAX_GEMINI_RETRIES):
            try:
                logging.info(f"Sending context to primary Gemini model ({GEMINI_MODEL_NAME}), attempt {attempt + 1}/{MAX_GEMINI_RETRIES} for {mentioned_post_uri}...")
                # Handle different versions of the Gemini SDK
                try:
                    # Attempt with direct JSON structure (following REST API example pattern)
                    logging.info(f"Attempt {attempt + 1}: Using REST API pattern with generationConfig")
                    
                    # Create a direct request structure similar to the REST API
                    request = {
                        "model": GEMINI_MODEL_NAME,
                        "contents": [{"parts": [{"text": full_prompt_for_gemini}]}],
                        "generationConfig": {
                            "responseModalities": ["TEXT", "IMAGE"],
                            "temperature": 0.7,
                            "maxOutputTokens": 800,
                            "topP": 0.95,
                            "topK": 64
                        }
                    }
                    
                    # Get raw client to send the request directly
                    raw_client = gemini_model_ref._client._api_client
                    response_json = raw_client.request(
                        'post', 
                        f'models/{GEMINI_MODEL_NAME}:generateContent',
                        request
                    )
                    
                    # Process raw response into expected format
                    logging.info(f"Received raw response: {response_json}")
                    primary_gemini_response_obj = gemini_model_ref._client._from_response(response_json)
                    
                except Exception as e1:
                    logging.warning(f"Direct REST API pattern failed: {str(e1)}")
                    try:
                        # Fallback to basic API call as last resort
                        logging.info(f"Attempt {attempt + 1}: Falling back to most basic API call")
                        primary_gemini_response_obj = gemini_model_ref.generate_content(
                            full_prompt_for_gemini
                        )
                    except Exception as e2:
                        logging.error(f"All API calling patterns failed in attempt {attempt + 1}. Errors: {str(e1)}, then {str(e2)}")
                        raise
                
                # Process text from the primary model
                if primary_gemini_response_obj.parts:
                    full_text_response = "".join(part.text for part in primary_gemini_response_obj.parts if hasattr(part, 'text'))
                    
                    # Check for IMAGE_PROMPT keyword
                    if "IMAGE_PROMPT:" in full_text_response:
                        parts = full_text_response.split("IMAGE_PROMPT:", 1)
                        gemini_response_text = parts[0].strip()
                        image_prompt_for_imagen = parts[1].strip()
                        logging.info(f"Attempt {attempt + 1}: Primary model provided text and an image prompt: '{image_prompt_for_imagen}'")
                    else:
                        gemini_response_text = full_text_response.strip()
                        logging.info(f"Attempt {attempt + 1}: Primary model provided text only.")
                else:
                    gemini_response_text = "" # Ensure it's an empty string if no parts

                # If we got usable text, break the retry loop for primary model
                if gemini_response_text or image_prompt_for_imagen:
                    logging.info(f"Attempt {attempt + 1}: Successfully got content from primary Gemini for {mentioned_post_uri}.")
                    break 
                else:
                    logging.warning(f"Attempt {attempt + 1}: Primary Gemini returned no usable text for {mentioned_post_uri}.")
                    if hasattr(primary_gemini_response_obj, 'prompt_feedback') and primary_gemini_response_obj.prompt_feedback:
                        logging.warning(f"Attempt {attempt + 1} Primary Gemini prompt feedback for {mentioned_post_uri}: {primary_gemini_response_obj.prompt_feedback}")
                    if hasattr(primary_gemini_response_obj, 'parts'):
                        logging.warning(f"Attempt {attempt + 1} Primary Gemini response parts for {mentioned_post_uri}: {primary_gemini_response_obj.parts}")
                    else:
                        logging.warning(f"Attempt {attempt + 1} Primary Gemini response object for {mentioned_post_uri} has no 'parts' attribute: {primary_gemini_response_obj}")
                    logging.warning(f"Attempt {attempt + 1} Full prompt sent to primary Gemini for {mentioned_post_uri}:\n{full_prompt_for_gemini}")

            except ValueError as ve: 
                logging.error(f"Attempt {attempt + 1}: Primary Gemini text generation failed for {mentioned_post_uri} (ValueError): {ve}")
                if hasattr(primary_gemini_response_obj, 'prompt_feedback') and primary_gemini_response_obj.prompt_feedback.block_reason:
                    logging.error(f"Attempt {attempt + 1}: Primary Gemini prompt blocked. Reason: {primary_gemini_response_obj.prompt_feedback.block_reason}")
                    if "block_reason" in str(ve).lower() or (hasattr(primary_gemini_response_obj, 'prompt_feedback') and primary_gemini_response_obj.prompt_feedback.block_reason):
                        return # Exit processing if definitively blocked by primary model
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}: Primary Gemini content generation failed for {mentioned_post_uri}: {e}", exc_info=True)
            
            if attempt < MAX_GEMINI_RETRIES - 1 and not (gemini_response_text or image_prompt_for_imagen):
                logging.info(f"Waiting {GEMINI_RETRY_DELAY_SECONDS}s before next primary Gemini attempt for {mentioned_post_uri}...")
                time.sleep(GEMINI_RETRY_DELAY_SECONDS)
        # End of primary Gemini retry loop

        # If primary model failed to produce any text or an image prompt, skip.
        if not gemini_response_text and not image_prompt_for_imagen:
            logging.error(f"All {MAX_GEMINI_RETRIES} attempts to get content from primary Gemini failed for {mentioned_post_uri}. Skipping reply.")
            return

        # --- Image Generation with Imagen, if requested by primary model ---
        if image_prompt_for_imagen and imagen_client: # Check if imagen_client is initialized
            for imagen_attempt in range(MAX_GEMINI_RETRIES): # Use same retry constants for now
                try:
                    logging.info(f"Sending prompt to Imagen model ({IMAGEN_MODEL_NAME}), attempt {imagen_attempt + 1}/{MAX_GEMINI_RETRIES} for image prompt: '{image_prompt_for_imagen}'")
                    
                    # Try direct REST API approach for image generation
                    try:
                        logging.info(f"Imagen Attempt {imagen_attempt + 1}: Using REST API pattern with generationConfig")
                        
                        # Create a direct request structure similar to the REST API
                        request = {
                            "model": IMAGEN_MODEL_NAME,
                            "contents": [{"parts": [{"text": image_prompt_for_imagen}]}],
                            "generationConfig": {
                                "responseModalities": ["TEXT", "IMAGE"],
                                "temperature": 0.7,
                                "maxOutputTokens": 800,
                                "topP": 0.95,
                                "topK": 64
                            }
                        }
                        
                        # Get raw client to send the request directly
                        raw_client = imagen_client._api_client
                        response_json = raw_client.request(
                            'post', 
                            f'models/{IMAGEN_MODEL_NAME}:generateContent',
                            request
                        )
                        
                        # Process raw response into expected format
                        logging.info(f"Received raw image response: {response_json}")
                        imagen_response = imagen_client.models._from_response(response_json)
                        
                    except Exception as e1:
                        logging.warning(f"Direct REST API pattern for image failed: {str(e1)}")
                        try:
                            # Fallback to basic API call as last resort
                            logging.info(f"Imagen Attempt {imagen_attempt + 1}: Falling back to most basic API call")
                            imagen_response = imagen_client.models.generate_content(
                                model=IMAGEN_MODEL_NAME,
                                contents=image_prompt_for_imagen
                            )
                        except Exception as e2:
                            logging.error(f"All image generation API calling patterns failed. Errors: {str(e1)}, then {str(e2)}")
                            raise
                    
                    # Extract the image bytes from the response
                    if imagen_response and imagen_response.candidates and len(imagen_response.candidates) > 0:
                        for part in imagen_response.candidates[0].content.parts:
                            if part.inline_data is not None:
                                image_data_bytes = part.inline_data.data
                                logging.info(f"Successfully generated image. Size: {len(image_data_bytes)} bytes")
                                generated_alt_text = image_prompt_for_imagen  # Use the prompt as alt text
                                break
                        if image_data_bytes:
                            break
                        else:
                            logging.warning(f"Imagen Attempt {imagen_attempt + 1}: Generated image response missing image data")
                    else:
                        logging.warning(f"Imagen Attempt {imagen_attempt + 1}: No images generated in response")

                except Exception as e:
                    logging.error(f"Imagen Attempt {imagen_attempt + 1}: Imagen generation failed: {e}", exc_info=True)

            if not image_data_bytes:
                logging.error(f"All {MAX_GEMINI_RETRIES} attempts to generate image with Imagen failed. Proceeding with text-only reply if available.")
        # --- End Image Generation ---
            
        # If only an image was requested by primary model but not generated, and no other text was provided by primary, don't post.
        if image_prompt_for_imagen and not image_data_bytes and not gemini_response_text:
            logging.warning(f"Primary model requested an image (prompt: '{image_prompt_for_imagen}') but Imagen failed, and no fallback text from primary. Skipping reply.")
            return
        
        # If no text at all (e.g. primary model only outputted IMAGE_PROMPT: and imagen failed), skip.
        if not gemini_response_text and not image_data_bytes:
            logging.warning(f"Neither text nor image could be generated for {mentioned_post_uri}. Skipping.")
            return

        # Fallback text if image was intended but failed, but primary model also gave text.
        if image_prompt_for_imagen and not image_data_bytes and gemini_response_text:
            gemini_response_text += "\n(Sorry, I tried to generate an image for you, but it didn't work out this time!) "
            logging.info("Image generation failed, but text response is available. Appending a note.")

        # Prepare post content (text, facets, embed)
        post_text = gemini_response_text.strip() if gemini_response_text else ""
        
        facets = []
        if post_text: 
            # Mentions (ensure DID resolution for production)
            mention_pattern = r'@([a-zA-Z0-9_.-]+(?:\.[a-zA-Z]+)?)' # Improved handle matching
            for match in re.finditer(mention_pattern, post_text):
                handle = match.group(1)
                # Calculate byte offsets correctly
                byte_start = len(post_text[:match.start()].encode('utf-8'))
                byte_end = len(post_text[:match.end()].encode('utf-8'))
                
                # Placeholder DID - resolve to actual DID in a real application
                # resolved_did = resolve_handle_to_did(handle) # You'd need this function
                # if resolved_did:
                facets.append(
                    at_models.AppBskyRichtextFacet.Main(
                        index=at_models.AppBskyRichtextFacet.ByteSlice(byteStart=byte_start, byteEnd=byte_end),
                        features=[at_models.AppBskyRichtextFacet.Mention(did=f"did:plc:{handle}")] # Replace with resolved_did
                    )
                )
            
            # Links (more robust regex)
            url_pattern = r'https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)'
            for match in re.finditer(url_pattern, post_text):
                uri = match.group(0) # group(0) gets the entire match
                # Validate if URI is well-formed before adding facet (optional, basic check here)
                if "://" in uri: # Simple check to ensure it looks like a protocol URI
                    # Calculate byte offsets correctly
                    byte_start = len(post_text[:match.start()].encode('utf-8'))
                    byte_end = len(post_text[:match.end()].encode('utf-8'))

                    facets.append(
                        at_models.AppBskyRichtextFacet.Main(
                            index=at_models.AppBskyRichtextFacet.ByteSlice(byteStart=byte_start, byteEnd=byte_end),
                            features=[at_models.AppBskyRichtextFacet.Link(uri=uri)]
                        )
                    )
        
        embed_to_post = None
        if image_data_bytes and bsky_client: # Ensure client is available
            try:
                # Skip extremely small images that are likely invalid
                if len(image_data_bytes) < 500:
                    logging.warning(f"Image data too small to be valid ({len(image_data_bytes)} bytes). Skipping image upload.")
                else:
                    logging.info(f"Uploading generated image to Bluesky... Size: {len(image_data_bytes)} bytes")
                    # Make sure we're working with raw bytes for upload_blob
                    # If image_data_bytes is already bytes, use it directly
                    # If it's base64 string, decode it first
                    if isinstance(image_data_bytes, str):
                        try:
                            image_data_bytes = base64.b64decode(image_data_bytes)
                            logging.info(f"Converted base64 string to bytes for upload. Length: {len(image_data_bytes)}")
                        except Exception as e:
                            logging.error(f"Error converting base64 string to bytes: {e}")
                    
                    response = bsky_client.com.atproto.repo.upload_blob(image_data_bytes)
                    
                    if response and hasattr(response, 'blob') and response.blob:
                        logging.info(f"Image uploaded successfully. Blob CID: {response.blob.cid}")
                        # Include more detailed logging of the blob object
                        if hasattr(response.blob, 'cid'):
                            logging.info(f"Blob CID: {response.blob.cid}")
                        if hasattr(response.blob, 'mimeType'):
                            logging.info(f"Blob MIME type: {response.blob.mimeType}")
                        
                        # Ensure alt text isn't too long (Bluesky may have limits)
                        if len(generated_alt_text) > 300:
                            generated_alt_text = generated_alt_text[:297] + "..."
                            logging.info(f"Truncated alt text to 300 chars: {generated_alt_text}")
                        
                        image_for_embed = at_models.AppBskyEmbedImages.Image(alt=generated_alt_text, image=response.blob)
                        embed_to_post = at_models.AppBskyEmbedImages.Main(images=[image_for_embed])
                    else:
                        logging.error("Failed to upload image or blob data missing in response.")
            except Exception as e:
                logging.error(f"Error uploading image to Bluesky: {e}", exc_info=True)

        # --- Determine Root and Parent for the Reply --- 
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
        logging.info(f"Sending reply to {mentioned_post_uri}: Text='{post_text[:50]}...', HasImage={bool(image_data_bytes)}") 
        
        # Make sure facets is None if empty, not an empty list, as per SDK expectations for some versions.
        facets_to_send = facets if facets else None

        try:
            logging.info(f"Post creation details: Root={root_ref.uri}, Parent={parent_ref.uri}, Text length={len(post_text)}, Embed present={embed_to_post is not None}")
            
            # More detailed logging of the parameters being sent
            if embed_to_post:
                logging.info(f"Embed type: {type(embed_to_post).__name__}")
                if hasattr(embed_to_post, 'images') and embed_to_post.images:
                    logging.info(f"Number of images in embed: {len(embed_to_post.images)}")
                    for idx, img in enumerate(embed_to_post.images):
                        logging.info(f"Image {idx+1} alt text: '{img.alt}'")
                        if hasattr(img, 'image') and img.image:
                            logging.info(f"Image {idx+1} blob info: type={type(img.image).__name__}, has cid={hasattr(img.image, 'cid')}")
            
            response = bsky_client.send_post(
                text=post_text,
                reply_to=at_models.AppBskyFeedPost.ReplyRef(root=root_ref, parent=parent_ref),
                embed=embed_to_post,
                facets=facets_to_send
            )
            
            logging.info(f"Post creation response: {response}")
            logging.info(f"Successfully sent reply to {mentioned_post_uri}")
        except Exception as post_error:
            logging.error(f"Error creating post: {post_error}", exc_info=True)

    except AtProtocolError as e:
        logging.error(f"Bluesky API error processing mention {mentioned_post_uri}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error processing mention {mentioned_post_uri}: {e}", exc_info=True)


def main_bot_loop():
    """Main loop for the bot to check for mentions and process them."""
    global bsky_client, gemini_model, imagen_client, processed_uris_this_run # Ensure access to initialized clients

    if not (bsky_client and gemini_model and imagen_client):
        logging.critical("Bluesky client, Gemini model, or Imagen client not initialized. Exiting main loop.")
        return
    
    logging.info("Bot starting main loop...")
    while True:
        try:
            # Fetch notifications - no seenAt parameter, rely on isRead and updateSeen
            params = ListNotificationsParams(limit=NOTIFICATION_FETCH_LIMIT)
            response = bsky_client.app.bsky.notification.list_notifications(params=params)

            if response and response.notifications:
                latest_notification_indexed_at_in_batch = None # To track for updateSeen
                
                logging.info(f"Fetched {len(response.notifications)} notifications.")

                # Sort by indexedAt to process older unread notifications first,
                # and to correctly find the latest for updateSeen
                sorted_notifications = sorted(response.notifications, key=lambda n: n.indexed_at)
                
                unread_count = sum(1 for n in sorted_notifications if not n.is_read)
                logging.info(f"Found {unread_count} unread notifications to process.")

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
                    
                    # Log the notification we're about to process
                    logging.info(f"Processing notification: type={notification.reason}, from={notification.author.handle}, at={notification.indexed_at}")
                        
                    if notification.reason in ['mention', 'reply']:
                        # Pass the gemini_model directly
                        process_mention(notification, gemini_model) 
                    else:
                        logging.debug(f"Skipping notification {notification.uri}: Reason '{notification.reason}' not 'mention' or 'reply'.")

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
    global bsky_client, gemini_model, imagen_client # Declare intent to modify globals

    logging.info("Bot starting...")
    
    bsky_client = initialize_bluesky_client()
    gemini_model = initialize_gemini_model()
    imagen_initialized = initialize_imagen_model() # Initialize Imagen client

    # No initial update_seen call based on a loaded timestamp
    if bsky_client and gemini_model and imagen_initialized: # Check all models
        main_bot_loop()
    else:
        if not bsky_client:
            logging.error("Failed to initialize Bluesky client. Bot cannot start.")
        if not gemini_model:
            logging.error("Failed to initialize Gemini model. Bot cannot start.")
        if not imagen_initialized:
            logging.error("Failed to initialize Imagen client. Bot cannot start.")

if __name__ == "__main__":
    main()