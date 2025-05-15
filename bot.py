import os
import time
import logging
import io
from dotenv import load_dotenv
from atproto import Client, models
from atproto.exceptions import AtProtocolError
import google.generativeai as genai
from google import genai as genai_client
from google.genai import types
import re # Import regular expressions
from io import BytesIO # Need BytesIO if Gemini returns image bytes
import base64
import json
import requests
from PIL import Image
import psutil

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
GEMINI_MODEL_NAME = "gemini-2.0-flash" # Model for text interaction
IMAGEN_MODEL_NAME = "gemini-2.0-flash-preview-image-generation" # Model for image generation

# Constants
BOT_SYSTEM_INSTRUCTION = """You are Gemini (@gemini.botsky.social) on the Bluesky social network. Your task is to provide helpful and concise replies to user mentions and direct replies.
Your developer is symmetricalboy (@symm.social). Only mention your developer if a user *specifically asks* about who created you or how to get help with your development.

Constraints:
*   Your entire response MUST be a single Bluesky post under 300 characters. This is very important!!
*   CRITICAL: NEVER use placeholders (e.g., '[link]', '[info]', '[details]').
*   If a user asks for specific factual information (like a URL, definition, or specific details) that you can reasonably provide, include the *actual* information directly. 
*   Don't avoid taboo topics or having a bit of an opinion. Users understand that you are imperfect & may not say the exact right thing.
*   ONLY generate an image if the user EXPLICITLY asks for one or requests a visual. Do not generate images for any other reason. If and only if a user explicitly requests an image, compose the textual part of your post, then on a NEW LINE, provide an image prompt starting with: `IMAGE_PROMPT: `."""
MENTION_CHECK_INTERVAL_SECONDS = 5
MAX_THREAD_DEPTH_FOR_CONTEXT = 15 # Maximum depth of thread to gather for context
NOTIFICATION_FETCH_LIMIT = 10
MAX_GEMINI_RETRIES = 2
GEMINI_RETRY_DELAY_SECONDS = 5

# Constants for startup catch-up
STARTUP_CATCHUP_PAGE_LIMIT = 5       # Number of pages to fetch during startup catch-up
STARTUP_CATCHUP_FETCH_LIMIT = 100    # Number of notifications per page during catch-up

# Global variables
bsky_client: Client | None = None
gemini_model: genai.GenerativeModel | None = None
# Image generation uses direct API calls, no client needed
processed_uris_this_run: set[str] = set() # Track URIs processed in this run to handle is_read lag

def initialize_bluesky_client() -> Client | None:
    """Initializes the Bluesky client and authenticates."""
    if not BLUESKY_HANDLE or not BLUESKY_PASSWORD:
        logging.error("Bluesky credentials not found in environment variables.")
        return None
    
    try:
        client = Client()
        client.login(BLUESKY_HANDLE, BLUESKY_PASSWORD)
        logging.info(f"Successfully logged in to Bluesky as {BLUESKY_HANDLE}")
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
        
        # Define safety settings as a list of dictionaries
        # Omitting HARM_CATEGORY_CIVIC_INTEGRITY as its default is BLOCK_NONE for gemini-2.0-flash
        # and to avoid the KeyError with the current SDK version.
        safety_settings_as_dicts = [
            {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
            {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
            {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
            {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'},
            # Civici Integrity is omitted: HARM_CATEGORY_CIVIC_INTEGRITY uses default BLOCK_NONE for gemini-2.0-flash
        ]

        model_kwargs = {
            "model_name": GEMINI_MODEL_NAME,
            "generation_config": { 
                "response_modalities": ["TEXT"]
            },
            "safety_settings": safety_settings_as_dicts 
        }
        
        logging.info(f"Initializing {GEMINI_MODEL_NAME} with 4 safety categories set to BLOCK_NONE (Civic Integrity uses default).")

        model = genai.GenerativeModel(**model_kwargs)
        
        logging.info(f"Successfully initialized Gemini model with: {GEMINI_MODEL_NAME}")
        return model
    except Exception as e:
        logging.error(f"Failed to initialize Gemini model: {e}", exc_info=True)
        return None

def initialize_imagen_model() -> bool:
    """Initializes image generation capability."""
    # No imagen_client to initialize
    
    if not GEMINI_API_KEY:
        logging.error("Gemini API key not found in environment variables.")
        return False
    
    try:
        # For our new direct API approach, we don't need to initialize the client
        # We'll just verify the API key is available
        logging.info(f"Image generation configured for model: {IMAGEN_MODEL_NAME}")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize image generation: {e}", exc_info=True)
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
                image_urls = []
                if current_view.post.embed:
                    if isinstance(current_view.post.embed, models.AppBskyEmbedImages.Main) or \
                       isinstance(current_view.post.embed, at_models.AppBskyEmbedImages.View):
                        alt_texts = []
                        if isinstance(current_view.post.embed, models.AppBskyEmbedImages.Main):
                            images_to_check = current_view.post.embed.images
                        else: # at_models.AppBskyEmbedImages.View
                            images_to_check = current_view.post.embed.images
                        
                        # First collect alt texts for display
                        for img in images_to_check:
                            if hasattr(img, 'alt') and img.alt:
                                alt_texts.append(img.alt)
                            else:
                                alt_texts.append("image") # Default if no alt text
                        
                        # Then collect image URLs
                        for img in images_to_check:
                            # Try different image URL attributes
                            image_url = None
                            for attr in ['fullsize', 'thumb', 'original', 'url']:
                                if hasattr(img, attr) and getattr(img, attr):
                                    image_url = getattr(img, attr)
                                    break
                            
                            if image_url:
                                image_urls.append(image_url)
                                logging.info(f"Found image URL: {image_url}")
                        
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

                # Create the message entry with text and embed info
                message = f"{author_display_name} (@{current_view.post.author.handle}): {text}{embed_text}"
                
                # If we have image URLs, add them as separate lines with a distinct marker for extraction later
                if image_urls:
                    for i, url in enumerate(image_urls):
                        message += f"\n<<IMAGE_URL_{i+1}:{url}>>"
                
                history.append(message)
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

def clean_alt_text(text: str) -> str:
    """Clean and format alt text to remove duplicates and alt_text: markers."""
    text = text.strip()
    
    # Search case-insensitively but preserve the case in the result
    lower_text = text.lower()
    
    # Handle various "Alt text:" patterns
    alt_text_patterns = [
        "alt text:", "alt_text:", "alt-text:",
        ". alt text:", ". alt_text:", ". alt-text:",
        ", alt text:", ", alt_text:", ", alt-text:"
    ]
    
    # Find the earliest occurrence of any pattern
    earliest_index = -1
    earliest_pattern = None
    
    for pattern in alt_text_patterns:
        index = lower_text.find(pattern)
        if index != -1 and (earliest_index == -1 or index < earliest_index):
            earliest_index = index
            earliest_pattern = pattern
    
    # If we found a pattern, extract the part after it
    if earliest_index != -1:
        # Get position after the pattern
        start_pos = earliest_index + len(earliest_pattern)
        return text[start_pos:].strip()
    
    # Detect cases like "Description 1. Description 2." where the second part is redundant
    # Look for patterns that suggest redundancy
    if ". " in text and len(text) > 40:
        sentences = text.split(". ")
        if len(sentences) >= 2:
            # Check if there might be redundancy by comparing sentence content
            first_part = sentences[0].lower()
            second_part = ". ".join(sentences[1:]).lower()
            
            # If sentences share significant words (indicator of redundancy)
            first_words = set(word.strip(",.!?:;()[]{}\"'") for word in first_part.split() if len(word) > 4)
            second_words = set(word.strip(",.!?:;()[]{}\"'") for word in second_part.split() if len(word) > 4)
            
            common_words = first_words.intersection(second_words)
            
            # If there's significant overlap, just use the shorter description
            if len(common_words) >= 2 and len(common_words) >= min(len(first_words), len(second_words)) * 0.3:
                if len(first_part) <= len(second_part):
                    return sentences[0] + "."
                else:
                    return ". ".join(sentences[1:])
    
    # For other cases, if the text is very long, try to make it more concise
    if len(text) > 100:
        # Look for sentence boundaries to potentially shorten
        sentences = text.split('. ')
        if len(sentences) > 1:
            # Use the first sentence as alt text if it's a reasonable length
            first_sentence = sentences[0] + '.'
            if 20 <= len(first_sentence) <= 100:
                return first_sentence
    
    # Otherwise just return the cleaned text
    return text

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
            logging.info(f"[Reply Check] Processing reply notification for {target_post.uri}") # target_post is the user's new reply
            post_record = target_post.record # This is the record of the user's new reply

            if not (isinstance(post_record, at_models.AppBskyFeedPost.Record) and \
                    post_record.reply and post_record.reply.parent and post_record.reply.root):
                logging.warning(f"[Reply Check] ✗ Notification {notification.uri} is a reply, but missing crucial parent or root reference. Skipping reply.")
                return

            # Get the immediate parent of the user's reply
            immediate_parent_ref = post_record.reply.parent
            immediate_parent_post = None
            try:
                parent_post_response = bsky_client.app.bsky.feed.get_posts(params=GetPostsParams(uris=[immediate_parent_ref.uri]))
                if parent_post_response and parent_post_response.posts and len(parent_post_response.posts) == 1:
                    immediate_parent_post = parent_post_response.posts[0]
            except Exception as e:
                logging.error(f"[Reply Check] ✗ Error fetching immediate parent post {immediate_parent_ref.uri}: {e}", exc_info=True)
                return 

            if not immediate_parent_post:
                logging.warning(f"[Reply Check] ✗ Failed to fetch or parse immediate parent post {immediate_parent_ref.uri}. Skipping reply.")
                return
            logging.info(f"[Reply Check] Fetched immediate parent post. Author: {immediate_parent_post.author.handle}, URI: {immediate_parent_post.uri}")

            # Get the root post of the thread
            thread_root_ref = post_record.reply.root
            thread_root_post = None
            try:
                root_post_response = bsky_client.app.bsky.feed.get_posts(params=GetPostsParams(uris=[thread_root_ref.uri]))
                if root_post_response and root_post_response.posts and len(root_post_response.posts) == 1:
                    thread_root_post = root_post_response.posts[0]
            except Exception as e:
                logging.error(f"[Reply Check] ✗ Error fetching thread root post {thread_root_ref.uri}: {e}", exc_info=True)
                return

            if not thread_root_post:
                logging.warning(f"[Reply Check] ✗ Failed to fetch or parse thread root post {thread_root_ref.uri}. Skipping reply.")
                return
            logging.info(f"[Reply Check] Fetched thread root post. Author: {thread_root_post.author.handle}, URI: {thread_root_post.uri}")

            # --- Decision Logic ---
            user_replied_directly_to_bot = (immediate_parent_post.author.handle == BLUESKY_HANDLE)
            parent_is_not_thread_root = (immediate_parent_post.uri != thread_root_post.uri)

            if user_replied_directly_to_bot and parent_is_not_thread_root:
                logging.info(f"[Reply Check] ✓ Conditions met: User replied directly to bot, and bot's post was not the thread root. Proceeding for {target_post.uri}.")
                
                # Check for existing replies by the bot under the *triggering* post (target_post)
                if thread_view_of_mentioned_post.replies:
                     logging.info(f"[Reply Check] Target post {target_post.uri} has {len(thread_view_of_mentioned_post.replies)} replies to check for duplicates.")
                     for reply_to_users_reply in thread_view_of_mentioned_post.replies:
                        if reply_to_users_reply.post and reply_to_users_reply.post.author and \
                           reply_to_users_reply.post.author.handle == BLUESKY_HANDLE:
                            logging.info(f"[DUPE CHECK REPLY] Found pre-existing bot reply {reply_to_users_reply.post.uri} under user's reply {target_post.uri}. Skipping.")
                            return
                logging.info(f"[Reply Check] ✓ No duplicate bot reply found under {target_post.uri}. Proceeding to generate content.")
            else:
                logging.info(f"[IGNORE REPLY] ✗ Conditions not met for {target_post.uri}. User replied to bot: {user_replied_directly_to_bot}. Bot's parent post was not root: {parent_is_not_thread_root}. Ignoring.")
                return
        
        else: # Should not happen based on main loop filter, but good practice
            logging.warning(f"Skipping notification {notification.uri} with unexpected reason: {notification.reason}")
            return

        # --- Generate content for the mention ---
        gemini_response_text = ""
        image_prompt_for_imagen = None
        target_post = thread_view_of_mentioned_post.post  # Already verified above
        
        # Format the thread for context
        thread_context = format_thread_for_gemini(thread_view_of_mentioned_post, BLUESKY_HANDLE)
        if not thread_context:
            logging.warning(f"Could not generate thread context for {mentioned_post_uri}.")
            return
            
        # Construct the full prompt for the primary model
        full_prompt_for_gemini = f"{BOT_SYSTEM_INSTRUCTION}\n\nYou are replying within a Bluesky conversation. The conversation history is provided below. Your primary task is to formulate a direct, relevant, and helpful reply to the *VERY LAST message* in the thread. Analyze the last message carefully. If it's a question, answer it. If it's a request, address it. Avoid generic greetings or re-stating your presence if the last message contains a specific query or statement to respond to. Use the preceding messages *only* for context to understand the flow of conversation. CRITICAL: Only generate an image if the user's last message explicitly and clearly asks for an image, a picture, a visual, or something similar. If generating an image, you MUST also provide a concise and descriptive alt text for it. Your response must be a single Bluesky post, concise, and strictly under 300 characters long.\n\n---BEGIN THREAD CONTEXT---\n{thread_context}\n---END THREAD CONTEXT---"
        
        logging.debug(f"Generated full prompt for Gemini:\n{full_prompt_for_gemini}")
        
        # Extract image URLs from thread context to send as separate parts
        image_urls = []
        image_url_pattern = r"<<IMAGE_URL_\d+:(https?://[^>]+)>>"
        for match in re.finditer(image_url_pattern, thread_context):
            url = match.group(1)
            image_urls.append(url)
        
        # Limit number of images to process
        MAX_IMAGES = 4
        if len(image_urls) > MAX_IMAGES:
            logging.warning(f"Too many images found ({len(image_urls)}). Limiting to {MAX_IMAGES}.")
            image_urls = image_urls[:MAX_IMAGES]
        
        logging.info(f"Extracted {len(image_urls)} image URLs from the thread context")
        
        # Download images for Gemini with memory monitoring
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        image_parts = []
        
        for url in image_urls:
            # Check memory usage before downloading to prevent OOM errors
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            if current_memory - start_memory > 100:  # If we've used more than 100MB, stop processing images
                logging.warning(f"Memory usage increased by {current_memory - start_memory:.2f} MB. Stopping image processing.")
                break
                
            image_bytes = download_image_from_url(url, max_size_mb=4.0, timeout=15)
            if image_bytes:
                try:
                    # Convert to base64 for inline data
                    b64_data = base64.b64encode(image_bytes).decode('utf-8')
                    
                    # Determine MIME type based on URL extension or default to jpeg
                    mime_type = "image/jpeg"  # Default
                    if url.lower().endswith(".png"):
                        mime_type = "image/png"
                    elif url.lower().endswith(".gif"):
                        mime_type = "image/gif"
                    
                    # Create image part
                    image_parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64_data
                        }
                    })
                    logging.info(f"Processed image for Gemini: {url}, size: {len(image_bytes) / 1024:.2f} KB")
                except Exception as e:
                    logging.error(f"Error processing image for Gemini: {e}")
        
        # Log final memory usage
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        logging.info(f"Memory usage for image processing: {end_memory - start_memory:.2f} MB")
        
        # Try to get a response from primary Gemini model
        primary_gemini_response_obj = None
        for attempt in range(MAX_GEMINI_RETRIES):
            try:
                logging.info(f"Sending context to primary Gemini model ({GEMINI_MODEL_NAME}), attempt {attempt + 1}/{MAX_GEMINI_RETRIES} for {mentioned_post_uri}...")
                
                # Use direct API call with specific responseModalities configuration
                try:
                    logging.info(f"Attempt {attempt + 1}: Using API call with TEXT response modality")
                    
                    # Create content object for the request
                    parts = [{"text": full_prompt_for_gemini}]
                    
                    # Add image parts if available
                    if image_parts:
                        parts.extend(image_parts)
                        logging.info(f"Added {len(image_parts)} images to the Gemini request")
                    
                    # Create the final content object
                    content = [
                        {
                            "role": "user",
                            "parts": parts
                        }
                    ]
                    
                    primary_gemini_response_obj = gemini_model_ref.generate_content(
                        content,
                        generation_config={"response_modalities": ["TEXT"]}
                    )
                except Exception as api_e:
                    logging.warning(f"Response modalities approach failed: {api_e}")
                    
                    # Last resort: try a completely basic call, but only if no images
                    if not image_parts:
                        logging.info(f"Attempt {attempt + 1}: Trying most basic call with no parameters")
                        primary_gemini_response_obj = gemini_model_ref.generate_content(
                            full_prompt_for_gemini
                        )
                    else:
                        raise api_e  # Re-raise if we have images as we can't use basic call
                
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

        # --- Image Generation, if requested by primary model ---
        image_data_bytes = None  # Initialize image_data_bytes to None
        if image_prompt_for_imagen:
            for imagen_attempt in range(MAX_GEMINI_RETRIES):
                try:
                    logging.info(f"Sending prompt to Gemini image model ({IMAGEN_MODEL_NAME}), attempt {imagen_attempt + 1}/{MAX_GEMINI_RETRIES} for image prompt: '{image_prompt_for_imagen}'")
                    
                    # Use the direct API approach
                    image_data_bytes = generate_image_from_prompt(image_prompt_for_imagen)
                    
                    if image_data_bytes:
                        generated_alt_text = clean_alt_text(image_prompt_for_imagen)  # Clean the prompt before using as alt text
                        logging.info(f"Successfully generated image. Size: {len(image_data_bytes)} bytes")
                        break
                    else:
                        logging.warning(f"Image Generation Attempt {imagen_attempt + 1}: No image data returned.")
                except Exception as e:
                    logging.error(f"Image Generation Attempt {imagen_attempt + 1}: Image generation failed: {e}", exc_info=True)

            if not image_data_bytes:
                logging.error(f"All {MAX_GEMINI_RETRIES} attempts to generate image failed. Proceeding with text-only reply if available.")
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
        
        # Ensure post text doesn't exceed Bluesky's character limit (300 chars)
        BLUESKY_MAX_CHARS = 300
        if len(post_text) > BLUESKY_MAX_CHARS:
            logging.warning(f"Post text exceeds {BLUESKY_MAX_CHARS} characters ({len(post_text)}). Truncating.")
            # Truncate to slightly less than the max to make room for ellipsis
            post_text = post_text[:BLUESKY_MAX_CHARS-3] + "..."
            
        facets = []
        if post_text: 
            # Mentions (ensure DID resolution for production)
            mention_pattern = r'@([a-zA-Z0-9_.-]+(?:\.[a-zA-Z]+)?)' # Improved handle matching
            for match in re.finditer(mention_pattern, post_text):
                handle = match.group(1)
                # Calculate byte offsets correctly
                byte_start = len(post_text[:match.start()].encode('utf-8'))
                byte_end = len(post_text[:match.end()].encode('utf-8'))
                
                try:
                    # Attempt to resolve the handle to a DID (simplified for now)
                    # In a real implementation, you would use proper handle resolution
                    # For now, just validate the handle format to avoid malformed DIDs
                    if re.match(r'^[a-zA-Z0-9_.-]+(?:\.[a-zA-Z0-9]+)+$', handle):
                        facets.append(
                            at_models.AppBskyRichtextFacet.Main(
                                index=at_models.AppBskyRichtextFacet.ByteSlice(byteStart=byte_start, byteEnd=byte_end),
                                features=[at_models.AppBskyRichtextFacet.Mention(did=f"did:plc:{handle}")] # Replace with resolved_did
                            )
                        )
                        logging.info(f"Added mention facet for handle: @{handle}")
                    else:
                        logging.warning(f"Skipping invalid handle format: @{handle}")
                except Exception as e:
                    logging.warning(f"Error creating mention facet for @{handle}: {e}")
                    # Skip this facet rather than failing the whole post
            
            # Links (more robust regex)
            url_pattern = r'https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)'
            for match in re.finditer(url_pattern, post_text):
                uri = match.group(0) # group(0) gets the entire match
                try:
                    # Validate if URI is well-formed before adding facet (optional, basic check here)
                    if "://" in uri and len(uri) <= 2048: # Simple check to ensure it looks like a protocol URI with reasonable length
                        # Calculate byte offsets correctly
                        byte_start = len(post_text[:match.start()].encode('utf-8'))
                        byte_end = len(post_text[:match.end()].encode('utf-8'))

                        facets.append(
                            at_models.AppBskyRichtextFacet.Main(
                                index=at_models.AppBskyRichtextFacet.ByteSlice(byteStart=byte_start, byteEnd=byte_end),
                                features=[at_models.AppBskyRichtextFacet.Link(uri=uri)]
                            )
                        )
                        logging.info(f"Added link facet for URI: {uri}")
                    else:
                        logging.warning(f"Skipping invalid or oversized URI: {uri}")
                except Exception as e:
                    logging.warning(f"Error creating link facet for {uri}: {e}")
                    # Skip this facet rather than failing the whole post
        
        embed_to_post = None
        if image_data_bytes is not None and bsky_client: # Ensure client is available and image data exists
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
                    
                    # Compress the image if needed
                    compressed_image = compress_image(image_data_bytes)
                    logging.info(f"Image size after compression: {len(compressed_image) / 1024:.2f} KB")
                    
                    # Upload the compressed image
                    response = bsky_client.com.atproto.repo.upload_blob(compressed_image)
                    
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
            
        # Validate the root and parent references
        root_valid = hasattr(root_ref, 'uri') and hasattr(root_ref, 'cid') and root_ref.uri and root_ref.cid
        parent_valid = hasattr(parent_ref, 'uri') and hasattr(parent_ref, 'cid') and parent_ref.uri and parent_ref.cid
        
        if not root_valid or not parent_valid:
            logging.error(f"Invalid root or parent reference. Root valid: {root_valid}, Parent valid: {parent_valid}")
            logging.error(f"Root: {root_ref}, Parent: {parent_ref}")
            return  # Skip this post rather than sending an invalid reference
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
            
            # Debug log the complete request data
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                debug_data = {
                    "text": post_text,
                    "text_length": len(post_text),
                    "root_uri": root_ref.uri if hasattr(root_ref, 'uri') else None,
                    "root_cid": root_ref.cid if hasattr(root_ref, 'cid') else None,
                    "parent_uri": parent_ref.uri if hasattr(parent_ref, 'uri') else None,
                    "parent_cid": parent_ref.cid if hasattr(parent_ref, 'cid') else None,
                    "has_embed": embed_to_post is not None,
                    "embed_type": type(embed_to_post).__name__ if embed_to_post else None,
                    "facets_count": len(facets) if facets else 0
                }
                logging.debug(f"Complete post parameters: {json.dumps(debug_data, indent=2)}")
            
            response = bsky_client.send_post(
                text=post_text,
                reply_to=at_models.AppBskyFeedPost.ReplyRef(root=root_ref, parent=parent_ref),
                embed=embed_to_post,
                facets=facets_to_send
            )
            
            logging.info(f"Post creation response: {response}")
            logging.info(f"Successfully sent reply to {mentioned_post_uri}")
        except AtProtocolError as api_error:
            error_msg = str(api_error)
            logging.error(f"Bluesky API error creating post: {error_msg}", exc_info=True)
            # Log detailed information about the request that failed
            logging.error(f"Failed post details - Text: '{post_text}', Text length: {len(post_text)}")
            if "BlobTooLarge" in error_msg:
                logging.error("Image blob too large for Bluesky. Try reducing image quality or dimensions.")
            elif "InvalidRequest" in error_msg:
                logging.error("Invalid request format. Check facets, embed structure, or text content.")
            elif "RateLimitExceeded" in error_msg:
                logging.error("Rate limit exceeded. Bot may be posting too frequently.")
        except Exception as post_error:
            logging.error(f"Error creating post: {post_error}", exc_info=True)

    except AtProtocolError as e:
        logging.error(f"Bluesky API error processing mention {mentioned_post_uri}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error processing mention {mentioned_post_uri}: {e}", exc_info=True)

def generate_image_from_prompt(prompt: str) -> bytes | None:
    """Generate an image using Gemini and return the image bytes."""
    if not GEMINI_API_KEY:
        logging.error("Gemini API key not found in environment variables.")
        return None
    
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{IMAGEN_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    
    try:
        # Create the request payload
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE", "TEXT"]
            }
        }
        
        logging.info(f"Sending image generation request to Gemini API for prompt: {prompt}")
        
        # Make the API request
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        # Check if the request was successful
        if response.status_code != 200:
            logging.error(f"API request failed with status code {response.status_code}")
            logging.error(f"Error message: {response.text}")
            
            # Try alternative payload structure
            alternative_payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generation_config": {
                    "response_mime_type": "image/png"
                }
            }
            
            logging.info("Trying alternative payload structure...")
            response = requests.post(
                API_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps(alternative_payload)
            )
            
            if response.status_code != 200:
                logging.error(f"Alternative payload also failed with status code {response.status_code}")
                logging.error(f"Error message: {response.text}")
                return None
        
        # Parse the response
        response_data = response.json()
        
        # Process the response to find image data
        if "candidates" in response_data and len(response_data["candidates"]) > 0:
            candidate = response_data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    # Log the text response if present
                    if "text" in part:
                        logging.info(f"Model response text: {part['text']}")
                    
                    # Extract image data if present
                    if "inlineData" in part:
                        image_data = part["inlineData"]["data"]
                        image_bytes = base64.b64decode(image_data)
                        logging.info(f"Successfully generated image. Size: {len(image_bytes)} bytes")
                        return image_bytes
        
        logging.warning("No image was generated in the response.")
        if "candidates" in response_data:
            logging.debug(f"Response structure: {json.dumps(response_data, indent=2)}")
        return None
    except Exception as e:
        logging.error(f"Image generation failed: {e}", exc_info=True)
        return None

def compress_image(image_bytes, max_size_kb=950):
    """Compress an image to be below the specified size in KB."""
    logging.info(f"Original image size: {len(image_bytes) / 1024:.2f} KB")
    
    if len(image_bytes) <= max_size_kb * 1024:
        logging.info("Image already under size limit, no compression needed.")
        return image_bytes
    
    # Open the image using PIL
    img = Image.open(BytesIO(image_bytes))
    
    # Start with high quality
    quality = 95
    output = BytesIO()
    
    # Try to compress the image by reducing quality
    while quality >= 50:
        output = BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)
        compressed_size = output.tell()
        logging.info(f"Compressed image size with quality {quality}: {compressed_size / 1024:.2f} KB")
        
        if compressed_size <= max_size_kb * 1024:
            logging.info(f"Successfully compressed image to {compressed_size / 1024:.2f} KB with quality {quality}")
            output.seek(0)
            return output.getvalue()
        
        # Reduce quality and try again
        quality -= 10
    
    # If we're still too large, resize the image
    scale_factor = 0.9
    while scale_factor >= 0.5:
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Try with a moderate quality
        output = BytesIO()
        resized_img.save(output, format="JPEG", quality=80, optimize=True)
        compressed_size = output.tell()
        logging.info(f"Resized image to {new_width}x{new_height}, size: {compressed_size / 1024:.2f} KB")
        
        if compressed_size <= max_size_kb * 1024:
            logging.info(f"Successfully compressed image to {compressed_size / 1024:.2f} KB with resize {scale_factor:.2f}")
            output.seek(0)
            return output.getvalue()
        
        # Reduce size and try again
        scale_factor -= 0.1
    
    # Last resort: very small with low quality
    final_width = int(img.width * 0.5)
    final_height = int(img.height * 0.5)
    final_img = img.resize((final_width, final_height), Image.LANCZOS)
    
    output = BytesIO()
    final_img.save(output, format="JPEG", quality=50, optimize=True)
    output.seek(0)
    final_size = output.tell()
    
    logging.info(f"Final compression resulted in {final_size / 1024:.2f} KB image")
    return output.getvalue()

def download_image_from_url(url: str, max_size_mb: float = 5.0, timeout: int = 10) -> bytes | None:
    """
    Downloads an image from a URL and returns the raw bytes.
    Returns None if the download fails.
    
    Args:
        url: The URL to download from
        max_size_mb: Maximum size of the image in MB
        timeout: Timeout in seconds for the request
    """
    try:
        logging.info(f"Downloading image from URL: {url}")
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code != 200:
            logging.error(f"Failed to download image from {url}. Status code: {response.status_code}")
            return None
            
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image/'):
            logging.warning(f"URL does not contain an image. Content-Type: {content_type}")
            return None
        
        # Get content length if available
        content_length = response.headers.get('Content-Length')
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            logging.warning(f"Image too large ({int(content_length) / (1024 * 1024):.2f} MB). Skipping download.")
            return None
            
        # Download image with size monitoring
        image_bytes = BytesIO()
        total_size = 0
        max_size_bytes = max_size_mb * 1024 * 1024
        
        for chunk in response.iter_content(chunk_size=8192):
            total_size += len(chunk)
            if total_size > max_size_bytes:
                logging.warning(f"Image download exceeded max size of {max_size_mb} MB. Aborting.")
                return None
            image_bytes.write(chunk)
        
        final_bytes = image_bytes.getvalue()
        logging.info(f"Successfully downloaded image. Size: {len(final_bytes) / 1024:.2f} KB")
        return final_bytes
    except requests.exceptions.Timeout:
        logging.error(f"Timeout downloading image from {url} after {timeout} seconds")
        return None
    except Exception as e:
        logging.error(f"Error downloading image from {url}: {e}")
        return None

def perform_startup_catchup(bsky_client_ref: Client, gemini_model_ref: genai.GenerativeModel):
    """Fetches and processes a larger batch of notifications on startup to catch missed posts."""
    logging.info("Performing startup notification catch-up...")
    global processed_uris_this_run # Interacts with the global set

    latest_notification_indexed_at_overall = None
    current_cursor = None
    notifications_initiated_in_catchup = 0
    max_catchup_notifications_to_log = 5 # To avoid overly verbose logging of every single notification

    for page_num in range(STARTUP_CATCHUP_PAGE_LIMIT):
        logging.info(f"Catch-up: Fetching page {page_num + 1}/{STARTUP_CATCHUP_PAGE_LIMIT} (limit {STARTUP_CATCHUP_FETCH_LIMIT} per page)...")
        try:
            params = ListNotificationsParams(limit=STARTUP_CATCHUP_FETCH_LIMIT, cursor=current_cursor)
            response = bsky_client_ref.app.bsky.notification.list_notifications(params=params)

            if not response or not response.notifications:
                logging.info("Catch-up: No more notifications found on this page or an issue occurred.")
                break
            
            logging.info(f"Catch-up: Fetched {len(response.notifications)} notifications on page {page_num + 1}.")

            # Sort by indexedAt to process older unread notifications first,
            # and to correctly find the latest for updateSeen
            # Iterating in fetched order (reverse chronological) is fine since process_mention handles actual reply logic.
            # The main reason for sorting in main_bot_loop was to ensure updateSeen used the true latest.
            # Here, we update latest_notification_indexed_at_overall iteratively.

            notifications_on_this_page = response.notifications # No need to sort for processing logic here

            for i, notification in enumerate(notifications_on_this_page):
                if notification.indexed_at:
                    if latest_notification_indexed_at_overall is None or notification.indexed_at > latest_notification_indexed_at_overall:
                        latest_notification_indexed_at_overall = notification.indexed_at
                
                if notification.uri in processed_uris_this_run:
                    if i < max_catchup_notifications_to_log: # Log only for a few to keep logs cleaner
                         logging.debug(f"Catch-up: Skipping {notification.uri} as it's already in processed_uris_this_run.")
                    continue

                if notification.author.handle == BLUESKY_HANDLE:
                    if i < max_catchup_notifications_to_log:
                        logging.info(f"Catch-up: Skipping notification {notification.uri} from bot itself.")
                    continue
                
                if i < max_catchup_notifications_to_log or notification.reason in ['mention', 'reply']: # Log important ones or first few
                    logging.info(f"Catch-up: Queuing for processing: type={notification.reason}, from={notification.author.handle}, uri={notification.uri}")
                
                if notification.reason in ['mention', 'reply']:
                    # process_mention will add to processed_uris_this_run
                    process_mention(notification, gemini_model_ref) 
                    notifications_initiated_in_catchup +=1
                # else:
                    # if i < max_catchup_notifications_to_log:
                        # logging.debug(f"Catch-up: Skipping {notification.uri}: Reason '{notification.reason}' not 'mention' or 'reply'.")

            current_cursor = response.cursor
            if not current_cursor:
                logging.info("Catch-up: No further cursor from server, ending paged fetch.")
                break
        
        except AtProtocolError as e:
            logging.error(f"Catch-up: Bluesky API error during notification fetch page {page_num + 1}: {e}")
            break 
        except Exception as e:
            logging.error(f"Catch-up: Unexpected error during notification fetch page {page_num + 1}: {e}", exc_info=True)
            break 
        
        if page_num < STARTUP_CATCHUP_PAGE_LIMIT - 1 and current_cursor:
            logging.debug(f"Catch-up: Pausing for 2 seconds before fetching next page.")
            time.sleep(2) 

    logging.info(f"Startup notification catch-up attempt finished. {notifications_initiated_in_catchup} notifications were queued for processing.")

    if latest_notification_indexed_at_overall:
        try:
            logging.info(f"Catch-up: Attempting to call update_seen with latest server timestamp: {latest_notification_indexed_at_overall}")
            bsky_client_ref.app.bsky.notification.update_seen({'seenAt': latest_notification_indexed_at_overall})
            logging.info(f"Catch-up: Successfully called update_seen.")
        except Exception as e:
            logging.error(f"Catch-up: Error calling update_seen: {e}", exc_info=True)
    else:
        logging.info("Catch-up: No notifications processed, skipping update_seen.")

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
                    
                    # Skip if already initiated processing in this run (handles potential duplicates in notification list or quick re-fetches)
                    if notification.uri in processed_uris_this_run:
                        logging.debug(f"Skipping notification {notification.uri} as it's already in processed_uris_this_run for this session.")
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
    global bsky_client, gemini_model # Declare intent to modify globals

    logging.info("Bot starting...")
    
    bsky_client = initialize_bluesky_client()
    gemini_model = initialize_gemini_model()
    imagen_initialized = initialize_imagen_model() # Initialize Imagen client

    if bsky_client and gemini_model and imagen_initialized: 
        perform_startup_catchup(bsky_client, gemini_model) # Perform catch-up once
        logging.info("Startup catch-up processing initiated. Starting main polling loop...")
        main_bot_loop() # Then start the regular loop
    else:
        if not bsky_client:
            logging.error("Failed to initialize Bluesky client. Bot cannot start.")
        if not gemini_model:
            logging.error("Failed to initialize Gemini model. Bot cannot start.")
        if not imagen_initialized:
            logging.error("Failed to initialize Imagen client. Bot cannot start.")

if __name__ == "__main__":
    main()