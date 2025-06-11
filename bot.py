#!/usr/bin/env python3
"""
Bluesky Bot with Google Gemini AI Integration

A sophisticated bot for the Bluesky social network that:
- Monitors real-time posts via Jetstream WebSocket API
- Processes mentions and replies using Google Gemini AI
- Generates images with Imagen 3 and videos with Veo 2
- Supports complex thread management and duplicate prevention
- Includes memory management and rate limiting

SECURITY NOTES:
- Uses BLOCK_NONE safety settings for Gemini API (maximum flexibility)
- Implements rate limiting for both Gemini and Bluesky APIs
- Thread-safe processing with proper locking mechanisms
- Critical errors are sent via DM to developer (@symm.social)

REQUIREMENTS:
- Python 3.8+
- Bluesky account with app password
- Google Gemini API key with access to required models
- Environment variables configured in .env file

DEPLOYMENT:
- Suitable for cloud deployment (Heroku, Railway, etc.)
- Memory monitoring and cleanup implemented
- Comprehensive logging for debugging

Author: symmetricalboy (@symm.social)
"""

import os
import time
import logging
import io
import collections
import urllib.parse
import asyncio
import json
import threading
from typing import Optional
from dotenv import load_dotenv
from atproto import Client, models
from atproto.exceptions import AtProtocolError
import google.genai as genai
from google.genai.types import Tool, GoogleSearch, GenerateVideosConfig
import re # Import regular expressions
from io import BytesIO # Need BytesIO if Gemini returns image bytes
import base64
import requests
from PIL import Image
import psutil
import websockets
import gc
from dataclasses import dataclass

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

# Validate required environment variables
def validate_environment_variables():
    """Validate that all required environment variables are present."""
    required_vars = [
        ("BLUESKY_HANDLE", "Bluesky handle/username"),
        ("BLUESKY_PASSWORD", "Bluesky app password"),
        ("GEMINI_API_KEY", "Google Gemini API key"),
        ("DEVELOPER_DID", "Developer DID for error notifications"),
        ("DEVELOPER_HANDLE", "Developer handle for error notifications")
    ]
    
    missing_vars = []
    for var_name, description in required_vars:
        if not os.getenv(var_name):
            missing_vars.append(f"{var_name} ({description})")
    
    if missing_vars:
        logging.critical(f"Missing required environment variables: {', '.join(missing_vars)}")
        logging.critical("Please check your .env file and ensure all required variables are set.")
        return False
    
    return True

# Validate environment on startup
if not validate_environment_variables():
    logging.critical("Cannot start bot due to missing environment variables.")
    exit(1)

BLUESKY_HANDLE = os.getenv("BLUESKY_HANDLE")
BLUESKY_PASSWORD = os.getenv("BLUESKY_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Environment Variables
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro-preview-06-05") # Model for text interaction
IMAGEN_MODEL_NAME = os.getenv("IMAGEN_MODEL_NAME", "imagen-3.0-generate-002") # Model for image generation
VEO_MODEL_NAME = os.getenv("VEO_MODEL_NAME", "veo-2.0-generate-001") # Model for video generation

# Media Generation Configuration (OPTIONAL - defaults provided)
VIDEO_PERSON_GENERATION = os.getenv("VIDEO_PERSON_GENERATION", "ALLOW_ADULT") # Video person generation: "ALLOW_ADULT", "ALLOW_MINOR", "dont_allow"
IMAGE_PERSON_GENERATION = os.getenv("IMAGE_PERSON_GENERATION", "ALLOW_ADULT") # Image person generation: "ALLOW_ADULT", "ALLOW_MINOR", "dont_allow"

# Text Generation Safety Settings (OPTIONAL - defaults provided)
# Options: "BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"
SAFETY_HARASSMENT = os.getenv("SAFETY_HARASSMENT", "BLOCK_NONE") # Harassment content threshold
SAFETY_HATE_SPEECH = os.getenv("SAFETY_HATE_SPEECH", "BLOCK_NONE") # Hate speech content threshold
SAFETY_SEXUALLY_EXPLICIT = os.getenv("SAFETY_SEXUALLY_EXPLICIT", "BLOCK_NONE") # Sexually explicit content threshold
SAFETY_DANGEROUS_CONTENT = os.getenv("SAFETY_DANGEROUS_CONTENT", "BLOCK_NONE") # Dangerous content threshold
SAFETY_CIVIC_INTEGRITY = os.getenv("SAFETY_CIVIC_INTEGRITY", "BLOCK_NONE") # Civic integrity content threshold

# Developer Configuration for Error Notifications (from environment)
DEVELOPER_DID = os.getenv("DEVELOPER_DID")
DEVELOPER_HANDLE = os.getenv("DEVELOPER_HANDLE")

# Bot Configuration from Environment Variables
MENTION_CHECK_INTERVAL_SECONDS = int(os.getenv("MENTION_CHECK_INTERVAL_SECONDS", "15")) # Default 60s is good for production
MAX_THREAD_DEPTH_FOR_CONTEXT = int(os.getenv("MAX_THREAD_DEPTH_FOR_CONTEXT", "25")) # Maximum depth of thread to gather for context
NOTIFICATION_FETCH_LIMIT = int(os.getenv("NOTIFICATION_FETCH_LIMIT", "25"))
MAX_GEMINI_RETRIES = int(os.getenv("MAX_GEMINI_RETRIES", "3"))
GEMINI_RETRY_DELAY_SECONDS = int(os.getenv("GEMINI_RETRY_DELAY_SECONDS", "15"))
MAX_VIDEO_GENERATION_RETRIES = int(os.getenv("MAX_VIDEO_GENERATION_RETRIES", "2")) # Number of retries for video generation
VIDEO_RETRY_DELAY_SECONDS = int(os.getenv("VIDEO_RETRY_DELAY_SECONDS", "30")) # Delay between video generation retries
MAX_IMAGE_GENERATION_RETRIES = int(os.getenv("MAX_IMAGE_GENERATION_RETRIES", "3")) # Number of retries for image generation
IMAGE_RETRY_DELAY_SECONDS = int(os.getenv("IMAGE_RETRY_DELAY_SECONDS", "10")) # Delay between image generation retries
CATCH_UP_NOTIFICATION_LIMIT = int(os.getenv("CATCH_UP_NOTIFICATION_LIMIT", "50")) # Number of notifications to check on startup for catch-up
MAX_REPLY_THREAD_DEPTH = int(os.getenv("MAX_REPLY_THREAD_DEPTH", "10")) # Max number of posts the bot will make in a thread
MAX_CONVERSATION_THREAD_DEPTH = int(os.getenv("MAX_CONVERSATION_THREAD_DEPTH", "50")) # Max number of posts in a convo before bot disengages
MAX_PROCESSED_URIS_CACHE = int(os.getenv("MAX_PROCESSED_URIS_CACHE", "500")) # Limit cache size to prevent memory leak

# Jetstream Configuration
JETSTREAM_ENDPOINT = os.getenv("JETSTREAM_ENDPOINT", "wss://jetstream2.us-west.bsky.network/subscribe")
JETSTREAM_RECONNECT_DELAY = int(os.getenv("JETSTREAM_RECONNECT_DELAY", "5")) # Seconds to wait before reconnecting

# Constants
BOT_SYSTEM_INSTRUCTION = f"""You are Gemini (@{BLUESKY_HANDLE}), a sophisticated AI assistant on the Bluesky social network. Your primary function is to provide accurate, professional, and educational responses to user inquiries.

**Core Directives:**
*   **Persona:** Maintain a professional, academic, and helpful tone. Avoid overly casual language, excessive emoji use, or flirtatious conversation. Your goal is to be a source of reliable information.
*   **Response Length & Threading:**
    *   For simple questions or brief interactions, provide a single, concise reply under 300 characters.
    *   For complex topics requiring a detailed explanation, you may generate a longer response. The hosting application will automatically split your response into a thread of up to 10 posts. Formulate your full response naturally, without worrying about the character limit of individual posts.
*   **Media Generation:**
    *   Only generate media (images or videos) when a user *explicitly* requests a visual or a generated asset. Do not offer or create media otherwise.
    *   Generate only ONE type of media per response (either an image or a video, not both).
    *   To trigger image generation, provide the textual part of your response, then on a NEW LINE, write `IMAGE_PROMPT: <prompt for image>`.
    *   To trigger video generation, provide the textual part of your response, then on a NEW LINE, write `VIDEO_PROMPT: <prompt for video>`.
*   **Factual Accuracy:** If a user asks for specific factual information (like a URL, definition, or data), provide the actual information directly. Do not use placeholders.
*   **Google Search Grounding Compliance:** If a user complains about or questions the "Grounded with Google Search" posts with search suggestions, politely explain that providing these search suggestions is a requirement from Google when using their grounding feature. Emphasize that you try to keep them as clean and contextually appropriate as possible, and they only appear when you've used Google Search to help answer their question.
*   **Developer Credit:** Only mention your developer, symmetricalboy (@symm.social), if a user specifically asks about your creation or development.
"""

# Global variables
bsky_client: Client | None = None
genai_client: genai.Client | None = None
processed_uris_this_run: collections.OrderedDict[str, None] = collections.OrderedDict() # Track URIs processed in this run
bot_did: str | None = None # Bot's DID for filtering

# Thread safety lock
_processed_uris_lock = threading.Lock()

# Rate limiting
@dataclass
class RateLimiter:
    last_gemini_call: float = 0.0
    last_bluesky_call: float = 0.0
    gemini_min_interval: float = 1.0  # Minimum 1 second between Gemini calls
    bluesky_min_interval: float = 0.5  # Minimum 0.5 seconds between Bluesky calls
    
    def wait_if_needed_gemini(self):
        current_time = time.time()
        time_since_last = current_time - self.last_gemini_call
        if time_since_last < self.gemini_min_interval:
            sleep_time = self.gemini_min_interval - time_since_last
            logging.info(f"Rate limiting: waiting {sleep_time:.2f}s before Gemini call")
            time.sleep(sleep_time)
        self.last_gemini_call = time.time()
    
    def wait_if_needed_bluesky(self):
        current_time = time.time()
        time_since_last = current_time - self.last_bluesky_call
        if time_since_last < self.bluesky_min_interval:
            sleep_time = self.bluesky_min_interval - time_since_last
            logging.info(f"Rate limiting: waiting {sleep_time:.2f}s before Bluesky call")
            time.sleep(sleep_time)
        self.last_bluesky_call = time.time()

rate_limiter = RateLimiter()

def is_content_policy_failure(error_msg: str, response_obj=None, prompt: str = None) -> bool:
    """Detect if a failure is due to content policy/safety filtering rather than technical issues."""
    if not error_msg:
        return False
    
    # Check for common content policy keywords in error messages
    policy_keywords = [
        "content policy", "safety", "blocked", "filtered", "person_generation",
        "inappropriate", "violates", "prohibited", "restricted", "harmful",
        "unsafe", "policy violation"
    ]
    
    error_lower = error_msg.lower()
    for keyword in policy_keywords:
        if keyword in error_lower:
            return True
    
    # Special case: API returned no videos/images but prompt contains people-related terms
    # This often indicates person_generation filtering
    if prompt and ("no videos" in error_lower or "no images" in error_lower):
        people_terms = ["person", "people", "human", "man", "woman", "child", "individual", "character"]
        prompt_lower = prompt.lower()
        for term in people_terms:
            if term in prompt_lower:
                return True
    
    # Check response object for policy-related feedback
    if response_obj and hasattr(response_obj, 'prompt_feedback'):
        if hasattr(response_obj.prompt_feedback, 'block_reason') and response_obj.prompt_feedback.block_reason:
            return True
    
    return False

def get_content_policy_message(media_type: str, prompt: str) -> str:
    """Generate a helpful message explaining content policy restrictions."""
    if media_type == "video":
        if "person" in prompt.lower() or "people" in prompt.lower() or "human" in prompt.lower():
            return "I can't generate videos with people in them due to content policy restrictions. Would you like me to try creating a video with a different concept?"
        else:
            return "I couldn't generate that video due to content policy restrictions. Could you try rephrasing your request?"
    elif media_type == "image":
        return "I couldn't generate that image due to content policy restrictions. Could you try a different description?"
    else:
        return "I couldn't generate that media due to content policy restrictions. Could you try a different approach?"

def send_developer_dm(error_message: str, error_type: str = "CRITICAL ERROR", allow_public_fallback: bool = False):
    """Send a DM to the developer about critical errors."""
    global bsky_client
    if not bsky_client:
        logging.error("Cannot send developer DM: Bluesky client not initialized")
        return False
    
    try:
        # Apply rate limiting
        rate_limiter.wait_if_needed_bluesky()
        
        # Truncate message if too long for DM
        max_dm_length = 1000
        if len(error_message) > max_dm_length:
            error_message = error_message[:max_dm_length-3] + "..."
        
        dm_text = f"🚨 {error_type}\n\nBot: @{BLUESKY_HANDLE}\nError: {error_message}\n\nTime: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
        
        # Create a chat client using the proxy
        try:
            dm_client = bsky_client.with_bsky_chat_proxy()
            dm = dm_client.chat.bsky.convo
            
            # Try to get existing conversation or create one
            convo = dm.get_convo_for_members(
                models.ChatBskyConvoGetConvoForMembers.Params(members=[DEVELOPER_DID])
            ).convo
            
            # Send the message
            dm.send_message(
                models.ChatBskyConvoSendMessage.Data(
                    convo_id=convo.id,
                    message=models.ChatBskyConvoDefs.MessageInput(
                        text=dm_text
                    )
                )
            )
            
            logging.info(f"✅ Sent developer DM about {error_type}")
            return True
            
        except Exception as dm_error:
            logging.error(f"Failed to send DM via chat API: {dm_error}")
            
            # Only fall back to public if explicitly allowed
            if allow_public_fallback:
                try:
                    fallback_text = f"@{DEVELOPER_HANDLE} 🚨 {error_type}: {error_message[:200]}..."
                    if len(fallback_text) > 300:
                        fallback_text = fallback_text[:297] + "..."
                    
                    facets = generate_facets_for_text(fallback_text, bsky_client)
                    bsky_client.send_post(
                        text=fallback_text,
                        facets=facets if facets else None
                    )
                    logging.info("Sent error notification as public mention (DM failed)")
                    return True
                except Exception as final_error:
                    logging.error(f"All notification methods failed: {final_error}")
                    return False
            else:
                logging.error(f"Failed to send DM and public fallback disabled")
                return False
        
    except Exception as outer_error:
        # Don't create infinite loops - just log the DM failure
        logging.error(f"Failed to send developer DM: {outer_error}")
        return False

def send_startup_notification(message: str):
    """Send a startup notification to the developer via DM only (no public fallback)."""
    success = send_developer_dm(message, "STARTUP NOTIFICATION", allow_public_fallback=False)
    if not success:
        logging.info(f"Startup notification could not be sent via DM: {message}")
        # For startup notifications, we just log it - don't spam public timeline
    return success

def log_critical_error(error_message: str, exception: Exception = None):
    """Log a critical error and notify the developer via DM."""
    if exception:
        full_error = f"{error_message}: {str(exception)}"
        logging.critical(full_error, exc_info=True)
    else:
        full_error = error_message
        logging.critical(full_error)
    
    # Send DM to developer (allow public fallback for critical errors)
    send_developer_dm(full_error, "CRITICAL ERROR", allow_public_fallback=True)

def initialize_bluesky_client() -> Client | None:
    """Initializes the Bluesky client and authenticates."""
    global bot_did
    if not BLUESKY_HANDLE or not BLUESKY_PASSWORD:
        logging.error("Bluesky credentials not found in environment variables.")
        return None
    
    try:
        client = Client()
        client.login(BLUESKY_HANDLE, BLUESKY_PASSWORD)
        
        # Store the bot's DID for filtering
        if hasattr(client, 'me') and client.me:
            bot_did = client.me.did
            logging.info(f"Bot DID: {bot_did}")
        
        logging.info(f"Successfully logged in to Bluesky as {BLUESKY_HANDLE}")
        return client
    except AtProtocolError as e:
        error_msg = f"Bluesky login failed: {e}"
        logging.error(error_msg)
        return None
    except Exception as e:
        error_msg = f"An unexpected error occurred during Bluesky login: {e}"
        logging.error(error_msg)
        return None

def initialize_genai_services() -> genai.Client | None:
    """Initializes the genai client."""
    if not GEMINI_API_KEY:
        logging.error("Gemini API key not found in environment variables.")
        return None
    
    try:
        # Create a client instance with the API key directly (new google-genai library)
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        logging.info(f"Successfully initialized GenAI Client.")
        logging.info(f"Text generation will use model: {GEMINI_MODEL_NAME} (max retries: {MAX_GEMINI_RETRIES})")
        logging.info(f"Text safety settings: Harassment={SAFETY_HARASSMENT}, Hate={SAFETY_HATE_SPEECH}, Sexual={SAFETY_SEXUALLY_EXPLICIT}, Dangerous={SAFETY_DANGEROUS_CONTENT}, Civic={SAFETY_CIVIC_INTEGRITY}")
        logging.info(f"Image generation configured for model: {IMAGEN_MODEL_NAME} (max retries: {MAX_IMAGE_GENERATION_RETRIES}, person_generation: {IMAGE_PERSON_GENERATION})")
        logging.info(f"Video generation configured for model: {VEO_MODEL_NAME} (max retries: {MAX_VIDEO_GENERATION_RETRIES}, person_generation: {VIDEO_PERSON_GENERATION})")
        return client
    except Exception as e:
        log_critical_error(f"Failed to initialize GenAI services", e)
        return None

def get_thread_length(thread_view: models.AppBskyFeedDefs.ThreadViewPost) -> int:
    """Recursively counts the number of posts in a thread chain."""
    count = 0
    current = thread_view
    while current:
        if isinstance(current, models.AppBskyFeedDefs.ThreadViewPost) and current.post:
            count += 1
        # Traverse up the parent chain
        if hasattr(current, 'parent') and current.parent:
            current = current.parent
        else:
            break
    return count

def split_text_for_bluesky(text: str, limit: int = 300) -> list[str]:
    """
    Splits a long string of text into a list of strings, each under the limit.
    Tries to split by sentences, then words, to make the posts readable.
    """
    if not text:
        return []

    posts = []
    # Use regex to split by sentences, keeping delimiters.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_post = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # If adding the next sentence exceeds the limit
        if len(current_post) + len(sentence) + 1 > limit:
            # If the current post has content, add it to the list
            if current_post:
                posts.append(current_post.strip())
            current_post = ""

            # If the sentence itself is over the limit, it needs to be split by words
            if len(sentence) > limit:
                words = sentence.split()
                word_post = ""
                for word in words:
                    if len(word_post) + len(word) + 1 > limit:
                        posts.append(word_post.strip())
                        word_post = word
                    else:
                        word_post += f" {word}"
                if word_post:
                    posts.append(word_post.strip())
            else:
                current_post = sentence
        else:
            if current_post:
                current_post += f" {sentence}"
            else:
                current_post = sentence

    if current_post:
        posts.append(current_post.strip())
        
    # Final check to ensure no post is empty
    return [post for post in posts if post]

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
                video_urls = []
                if current_view.post.embed:
                    logging.info(f"EMBED DETECTED: {type(current_view.post.embed)}")
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
                         isinstance(current_view.post.embed, at_models.AppBskyEmbedVideo.View):
                        # Extract video URLs from blob references
                        logging.info(f"🎥 VIDEO EMBED DETECTED: {type(current_view.post.embed)}")
                        logging.info(f"🎥 VIDEO EMBED ATTRIBUTES: {dir(current_view.post.embed)}")
                        if hasattr(current_view.post.embed, '__dict__'):
                            logging.info(f"🎥 VIDEO EMBED DICT: {current_view.post.embed.__dict__}")
                        
                        videos_to_check = []
                        if isinstance(current_view.post.embed, models.AppBskyEmbedVideo.Main):
                            logging.info(f"🎥 Checking Main embed for video attribute...")
                            if hasattr(current_view.post.embed, 'video'):
                                videos_to_check = [current_view.post.embed.video]
                                logging.info(f"🎥 VIDEO OBJECT FOUND in Main embed")
                            else:
                                logging.warning(f"🎥 Main embed has NO video attribute!")
                        else: # at_models.AppBskyEmbedVideo.View
                            logging.info(f"🎥 Processing View embed - video data is directly on embed object")
                            # For View embeds, the video data is directly on the embed object
                            videos_to_check = [current_view.post.embed]
                            logging.info(f"🎥 Using View embed object directly as video source")
                        
                        # Collect video URLs from blob references
                        for vid in videos_to_check:
                            logging.info(f"PROCESSING VIDEO OBJECT: {vid}")
                            logging.info(f"VIDEO OBJECT TYPE: {type(vid)}")
                            logging.info(f"VIDEO OBJECT ATTRIBUTES: {dir(vid)}")
                            if hasattr(vid, '__dict__'):
                                logging.info(f"VIDEO OBJECT DICT: {vid.__dict__}")
                            
                            # Video blobs need to be downloaded via PDS getBlob endpoint
                            # We'll construct the video URL using the blob CID
                            blob_cid = None
                            
                            # Try multiple approaches to get the blob CID
                            if hasattr(vid, 'ref'):
                                logging.info(f"VIDEO HAS REF: {vid.ref}")
                                if hasattr(vid.ref, '$link'):
                                    blob_cid = getattr(vid.ref, '$link')
                                    logging.info(f"FOUND BLOB CID via ref.$link: {blob_cid}")
                                elif hasattr(vid.ref, 'link'):
                                    blob_cid = vid.ref.link
                                    logging.info(f"FOUND BLOB CID via ref.link: {blob_cid}")
                                elif hasattr(vid.ref, '__dict__'):
                                    logging.info(f"REF DICT: {vid.ref.__dict__}")
                                    blob_cid = vid.ref.__dict__.get('link') or vid.ref.__dict__.get('$link')
                                    if blob_cid:
                                        logging.info(f"FOUND BLOB CID via ref dict: {blob_cid}")
                            
                            # Alternative: direct CID attribute
                            if not blob_cid and hasattr(vid, 'cid'):
                                blob_cid = vid.cid
                                logging.info(f"FOUND BLOB CID via direct cid: {blob_cid}")
                            
                            # Try to get CID from other possible attributes
                            if not blob_cid:
                                for attr in ['blob', 'video_blob', 'content', 'data']:
                                    if hasattr(vid, attr):
                                        obj = getattr(vid, attr)
                                        logging.info(f"CHECKING ATTR {attr}: {obj}")
                                        if hasattr(obj, 'ref') and hasattr(obj.ref, '$link'):
                                            blob_cid = getattr(obj.ref, '$link')
                                            logging.info(f"FOUND BLOB CID via {attr}.ref.$link: {blob_cid}")
                                            break
                                        elif hasattr(obj, 'cid'):
                                            blob_cid = obj.cid
                                            logging.info(f"FOUND BLOB CID via {attr}.cid: {blob_cid}")
                                            break
                            
                            if blob_cid:
                                video_urls.append(f"BLOB:{blob_cid}")
                                logging.info(f"SUCCESSFULLY EXTRACTED VIDEO BLOB CID: {blob_cid}")
                            else:
                                logging.warning(f"NO BLOB CID FOUND in video object after all attempts")
                        
                        # Get alt text for video
                        video_alt = ""
                        if isinstance(current_view.post.embed, models.AppBskyEmbedVideo.Main):
                            if hasattr(current_view.post.embed, 'alt') and current_view.post.embed.alt:
                                video_alt = current_view.post.embed.alt
                        else: # at_models.AppBskyEmbedVideo.View
                            if hasattr(current_view.post.embed, 'alt') and current_view.post.embed.alt:
                                video_alt = current_view.post.embed.alt
                        
                        if video_alt:
                            embed_text = f" [User attached video: {video_alt}]"
                        else:
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
                
                # If we have video URLs, add them as separate lines with a distinct marker for extraction later
                if video_urls:
                    for i, url in enumerate(video_urls):
                        message += f"\n<<VIDEO_URL_{i+1}:{url}>>"
                
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

def resolve_handle_to_did(handle: str, client: Client) -> str | None:
    """Resolves a Bluesky handle to its corresponding DID."""
    try:
        # Use the resolve_handle method from the AT Protocol client
        result = client.com.atproto.identity.resolve_handle({'handle': handle})
        if result and hasattr(result, 'did'):
            logging.debug(f"Resolved handle @{handle} to DID: {result.did}")
            return result.did
        else:
            logging.warning(f"Failed to resolve handle @{handle}: No DID in response")
            return None
    except Exception as e:
        logging.warning(f"Error resolving handle @{handle} to DID: {e}")
        return None

def generate_facets_for_text(text: str, client: Client) -> list:
    """Generates facets for mentions and links in the given text."""
    facets = []
    if not text:
        return facets
    
    # Handle mentions
    mention_pattern = r'@([a-zA-Z0-9_.-]+(?:\.[a-zA-Z0-9_.-]+)*\.(?:[a-zA-Z]{2,}|[a-zA-Z0-9_.-]+))'
    for match in re.finditer(mention_pattern, text):
        handle = match.group(1)
        byte_start = len(text[:match.start()].encode('utf-8'))
        byte_end = len(text[:match.end()].encode('utf-8'))
        try:
            resolved_did = resolve_handle_to_did(handle, client)
            if resolved_did:
                facets.append(
                    at_models.AppBskyRichtextFacet.Main(
                        index=at_models.AppBskyRichtextFacet.ByteSlice(byteStart=byte_start, byteEnd=byte_end),
                        features=[at_models.AppBskyRichtextFacet.Mention(did=resolved_did)]
                    )
                )
            else:
                logging.warning(f"Could not resolve handle @{handle} to DID")
        except Exception as e:
            logging.warning(f"Error creating mention facet for @{handle}: {e}")
    
    # Handle links
    url_pattern = r'https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)'
    for match in re.finditer(url_pattern, text):
        uri = match.group(0)
        try:
            if "://" in uri and len(uri) <= 2048:
                byte_start = len(text[:match.start()].encode('utf-8'))
                byte_end = len(text[:match.end()].encode('utf-8'))
                facets.append(
                    at_models.AppBskyRichtextFacet.Main(
                        index=at_models.AppBskyRichtextFacet.ByteSlice(byteStart=byte_start, byteEnd=byte_end),
                        features=[at_models.AppBskyRichtextFacet.Link(uri=uri)]
                    )
                )
        except Exception as e:
            logging.warning(f"Error creating link facet for {uri}: {e}")
    
    return facets

def clean_alt_text(text: str) -> str:
    """Clean and format alt text to remove duplicates and alt_text: markers."""
    text = text.strip()
    
    # Search case-insensitively but preserve the case in the result
    lower_text = text.lower()
    
    # Handle various "Alt text:" patterns
    alt_text_patterns = [
        "alt text:", "alt_text:", "alt-text:", "alt:",
        ". alt text:", ". alt_text:", ". alt-text:", ". alt:",
        ", alt text:", ", alt_text:", ", alt-text:", ", alt:"
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

def process_mention(notification: at_models.AppBskyNotificationListNotifications.Notification, genai_client_ref: genai.Client):
    """Processes a single mention/reply notification."""
    global bsky_client, processed_uris_this_run # Ensure globals are accessible
    if not bsky_client:
        logging.error("Bluesky client not initialized in process_mention. Cannot process mention.")
        return

    mentioned_post_uri = notification.uri
    # Thread-safe marking as seen for this run *before* any processing attempts to prevent loops if is_read lag
    with _processed_uris_lock:
        processed_uris_this_run[mentioned_post_uri] = None
        if len(processed_uris_this_run) > MAX_PROCESSED_URIS_CACHE:
            processed_uris_this_run.popitem(last=False) # Evict the oldest item
    
    logging.debug(f"Processing mention/reply in post: {mentioned_post_uri}")

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

        # --- Anti-Looping / Thread Depth Check ---
        thread_length = get_thread_length(thread_view_of_mentioned_post)
        logging.debug(f"Current conversation thread length is {thread_length}.")
        if thread_length >= MAX_CONVERSATION_THREAD_DEPTH:
            logging.info(f"🔁 Thread too long ({thread_length}), sending limit message.")
            try:
                # We still need root/parent info to reply
                canned_parent_ref = at_models.ComAtprotoRepoStrongRef.Main(cid=target_post.cid, uri=target_post.uri)
                canned_root_ref = None
                post_record = target_post.record
                if isinstance(post_record, at_models.AppBskyFeedPost.Record) and post_record.reply:
                    canned_root_ref = post_record.reply.root
                if canned_root_ref is None:
                    canned_root_ref = canned_parent_ref
                
                bsky_client.send_post(
                    text="Sorry, this thread has gotten very long! If you'd like to continue our conversation, please start a new thread.",
                    reply_to=at_models.AppBskyFeedPost.ReplyRef(root=canned_root_ref, parent=canned_parent_ref)
                )
                logging.info("✅ Sent thread depth limit message.")
            except Exception as e:
                logging.error(f"Failed to send thread depth limit message: {e}")
            return # Stop processing this mention
            
        # --- Reason-Specific Logic ---
        if notification.reason == 'mention':
            logging.debug(f"[Mention Check] Processing mention in {target_post.uri}")
            # Check for existing replies by the bot under the mentioned post (target_post)
            if thread_view_of_mentioned_post.replies:
                 for reply_in_thread in thread_view_of_mentioned_post.replies:
                    if reply_in_thread.post and reply_in_thread.post.author and reply_in_thread.post.author.handle == BLUESKY_HANDLE:
                        logging.debug(f"[DUPE CHECK MENTION] Found pre-existing bot reply {reply_in_thread.post.uri} to mentioned post {target_post.uri}. Skipping.")
                        return
            # If no duplicate found, fall through to generate context and reply...
            logging.debug(f"[Mention Check] No duplicate bot reply found for mention {target_post.uri}. Proceeding.")

        elif notification.reason == 'reply':
            logging.debug(f"[Reply Check] Processing reply notification for {target_post.uri}")
            post_record = target_post.record
            # Check if the target post is a valid reply with parent info
            if isinstance(post_record, at_models.AppBskyFeedPost.Record) and post_record.reply and post_record.reply.parent:
                parent_ref = post_record.reply.parent
                logging.debug(f"[Reply Check] Post {target_post.uri} replies to parent URI: {parent_ref.uri}. Fetching parent...")
                try:
                    get_parent_params = GetPostsParams(uris=[parent_ref.uri])
                    parent_post_response = bsky_client.app.bsky.feed.get_posts(params=get_parent_params)
                    
                    # Add detailed logging about the parent post response
                    if parent_post_response:
                        logging.debug(f"[Reply Check] Parent post response received. Has posts: {bool(parent_post_response.posts)}, Posts count: {len(parent_post_response.posts) if parent_post_response.posts else 0}")
                    
                    if parent_post_response and parent_post_response.posts and len(parent_post_response.posts) == 1:
                        immediate_parent_post = parent_post_response.posts[0]
                        logging.debug(f"[Reply Check] Fetched immediate parent post. Author: {immediate_parent_post.author.handle}, URI: {immediate_parent_post.uri}")

                        # **REVISED LOGIC**: Only proceed if the immediate parent IS the bot.
                        if immediate_parent_post.author.handle == BLUESKY_HANDLE:
                            logging.debug(f"[Reply Check] ✓ Immediate parent is the bot. Continue processing.")
                            logging.debug(f"[Reply Check] Checking for duplicate replies under {target_post.uri}...")
                            # Check for existing replies by the bot under the *triggering* post (target_post)
                            if thread_view_of_mentioned_post.replies:
                                 logging.debug(f"[Reply Check] Target post has {len(thread_view_of_mentioned_post.replies)} replies to check for duplicates.")
                                 for reply_to_users_reply in thread_view_of_mentioned_post.replies:
                                    if reply_to_users_reply.post and reply_to_users_reply.post.author and \
                                       reply_to_users_reply.post.author.handle == BLUESKY_HANDLE:
                                        logging.debug(f"[DUPE CHECK REPLY] Found pre-existing bot reply {reply_to_users_reply.post.uri} under user's reply {target_post.uri}. Skipping.")
                                        return
                            # If no duplicate found, fall through to generate context and reply...
                            logging.debug(f"[Reply Check] ✓ No duplicate bot reply found under {target_post.uri}. Proceeding.")
                        else:
                            # Parent is another user, ignore this reply.
                            logging.debug(f"[IGNORE USER-TO-USER REPLY] ✗ Notification {notification.uri} is a reply to another user ({immediate_parent_post.author.handle}), not the bot. Ignoring.")
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

        # --- Generate content for the mention ---
        gemini_response_text = ""
        image_prompt_for_imagen = None
        video_prompt = None
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
        
        # Extract video URLs from thread context to send as separate parts
        video_urls = []
        video_url_pattern = r"<<VIDEO_URL_\d+:(BLOB:[^>]+|https?://[^>]+)>>"
        for match in re.finditer(video_url_pattern, thread_context):
            url = match.group(1)
            video_urls.append(url)
        
        if video_urls:
            logging.info(f"🎥 EXTRACTED {len(video_urls)} video URLs: {video_urls}")
        else:
            logging.debug(f"No video URLs extracted")
        
        # Limit number of images to process
        MAX_IMAGES = 4
        if len(image_urls) > MAX_IMAGES:
            logging.warning(f"Too many images found ({len(image_urls)}). Limiting to {MAX_IMAGES}.")
            image_urls = image_urls[:MAX_IMAGES]
        
        # Limit number of videos to process
        MAX_VIDEOS = 2  # Videos are larger, so process fewer
        if len(video_urls) > MAX_VIDEOS:
            logging.warning(f"Too many videos found ({len(video_urls)}). Limiting to {MAX_VIDEOS}.")
            video_urls = video_urls[:MAX_VIDEOS]
        
        if image_urls or video_urls:
            logging.info(f"📎 Found {len(image_urls)} images, {len(video_urls)} videos in context")
        else:
            logging.debug(f"No media found in thread context")
        
        # Download images for Gemini with memory monitoring
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        image_parts = []
        downloaded_media_data = []  # Track downloaded data for cleanup
        
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
        
        # Download videos for Gemini with memory monitoring
        video_parts = []
        
        for url in video_urls:
            # Check memory usage before downloading to prevent OOM errors
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            if current_memory - start_memory > 200:  # If we've used more than 200MB total, stop processing videos
                logging.warning(f"Memory usage increased by {current_memory - start_memory:.2f} MB. Stopping video processing.")
                break
            
            video_bytes = None
            if url.startswith("BLOB:"):
                # Handle blob CID - need to get it from the PDS
                blob_cid = url[5:]  # Remove "BLOB:" prefix
                
                # We need to get the author's PDS endpoint from the thread context
                # For now, we'll try to extract it from the current post being processed
                try:
                    # Get the post author's DID and PDS endpoint
                    target_post = thread_view_of_mentioned_post.post
                    author_did = target_post.author.did
                    
                    # Try to resolve the author's handle to get PDS info
                    author_handle = target_post.author.handle
                    try:
                        # Use bsky_client to resolve handle to DID and get repo info
                        repo_desc = bsky_client.com.atproto.repo.describe_repo({'repo': author_did})
                        if repo_desc and hasattr(repo_desc, 'service'):
                            pds_endpoint = repo_desc.service
                            video_bytes = download_video_blob_from_pds(blob_cid, pds_endpoint, author_did)
                        else:
                            # Fallback: assume same PDS as the bot (works for posts from bsky.social)
                            pds_endpoint = "https://bsky.social"
                            video_bytes = download_video_blob_from_pds(blob_cid, pds_endpoint, author_did)
                    except Exception as resolve_error:
                        logging.warning(f"Could not resolve PDS for author {author_handle}, trying bsky.social: {resolve_error}")
                        # Fallback: assume bsky.social PDS
                        pds_endpoint = "https://bsky.social"
                        video_bytes = download_video_blob_from_pds(blob_cid, pds_endpoint, author_did)
                        
                except Exception as e:
                    logging.error(f"Error resolving author PDS for video blob {blob_cid}: {e}")
                    continue
            else:
                # Handle regular HTTP URL
                video_bytes = download_video_from_url(url, max_size_mb=15.0, timeout=30)
            
            if video_bytes:
                try:
                    # Convert to base64 for inline data
                    b64_data = base64.b64encode(video_bytes).decode('utf-8')
                    
                    # Determine MIME type based on URL extension or default to mp4
                    mime_type = "video/mp4"  # Default
                    if url.lower().endswith(".webm"):
                        mime_type = "video/webm"
                    elif url.lower().endswith(".mov"):
                        mime_type = "video/quicktime"
                    elif url.lower().endswith(".avi"):
                        mime_type = "video/x-msvideo"
                    
                    # Create video part
                    video_parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64_data
                        }
                    })
                    logging.info(f"Processed video for Gemini: {url}, size: {len(video_bytes) / 1024:.2f} KB")
                except Exception as e:
                    logging.error(f"Error processing video for Gemini: {e}")
        
        # Log final memory usage after video processing
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        logging.info(f"Total memory usage for media processing: {final_memory - start_memory:.2f} MB")
        
        # Try to get a response from primary Gemini model
        primary_gemini_response_obj = None
        for attempt in range(MAX_GEMINI_RETRIES):
            try:
                logging.info(f"🤖 Sending to Gemini (attempt {attempt + 1}/{MAX_GEMINI_RETRIES})")
                
                # Apply rate limiting
                rate_limiter.wait_if_needed_gemini()
                
                # Create content object for the request
                parts = [{"text": full_prompt_for_gemini}]
                
                # Add image parts if available
                if image_parts:
                    parts.extend(image_parts)
                    logging.debug(f"Added {len(image_parts)} images to the Gemini request")
                
                # Add video parts if available
                if video_parts:
                    parts.extend(video_parts)
                    logging.info(f"📹 Added {len(video_parts)} videos to the Gemini request")
                
                content = [{"role": "user", "parts": parts}]

                # Configure the tool for the API call
                google_search_tool = Tool(google_search=GoogleSearch())

                # Use the new google-genai library API
                primary_gemini_response_obj = genai_client_ref.models.generate_content(
                    model=GEMINI_MODEL_NAME,
                    contents=content,
                    config=genai.types.GenerateContentConfig(
                        tools=[google_search_tool],
                        max_output_tokens=20000,
                        safety_settings=[
                            genai.types.SafetySetting(
                                category='HARM_CATEGORY_HARASSMENT',
                                threshold=SAFETY_HARASSMENT
                            ),
                            genai.types.SafetySetting(
                                category='HARM_CATEGORY_HATE_SPEECH',
                                threshold=SAFETY_HATE_SPEECH
                            ),
                            genai.types.SafetySetting(
                                category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                                threshold=SAFETY_SEXUALLY_EXPLICIT
                            ),
                            genai.types.SafetySetting(
                                category='HARM_CATEGORY_DANGEROUS_CONTENT',
                                threshold=SAFETY_DANGEROUS_CONTENT
                            ),
                            genai.types.SafetySetting(
                                category='HARM_CATEGORY_CIVIC_INTEGRITY',
                                threshold=SAFETY_CIVIC_INTEGRITY
                            ),
                        ]
                    )
                )
                
                # Process text from the primary model
                if primary_gemini_response_obj.candidates and primary_gemini_response_obj.candidates[0].content.parts:
                    full_text_response = "".join(part.text for part in primary_gemini_response_obj.candidates[0].content.parts if hasattr(part, 'text'))
                    
                    video_prompt = None
                    if "VIDEO_PROMPT:" in full_text_response:
                        parts = full_text_response.split("VIDEO_PROMPT:", 1)
                        gemini_response_text = parts[0].strip()
                        video_prompt = parts[1].strip()
                        image_prompt_for_imagen = None
                        logging.info(f"Attempt {attempt + 1}: Primary model provided text and a video prompt: '{video_prompt}'")
                    elif "IMAGE_PROMPT:" in full_text_response:
                        parts = full_text_response.split("IMAGE_PROMPT:", 1)
                        gemini_response_text = parts[0].strip()
                        image_prompt_for_imagen = parts[1].strip()
                        logging.info(f"Attempt {attempt + 1}: Primary model provided text and an image prompt: '{image_prompt_for_imagen}'")
                    else:
                        gemini_response_text = full_text_response.strip()
                        image_prompt_for_imagen = None
                        logging.info(f"Attempt {attempt + 1}: Primary model provided text only.")
                else:
                    gemini_response_text = "" # Ensure it's an empty string if no parts
                    image_prompt_for_imagen = None
                    video_prompt = None

                # If we got usable text, break the retry loop for primary model
                if gemini_response_text or image_prompt_for_imagen or video_prompt:
                    logging.info(f"✅ Got response from Gemini (attempt {attempt + 1})")
                    break 
                else:
                    logging.warning(f"⚠️ Gemini returned no usable content (attempt {attempt + 1})")
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
            
            if attempt < MAX_GEMINI_RETRIES - 1 and not (gemini_response_text or image_prompt_for_imagen or video_prompt):
                logging.info(f"Waiting {GEMINI_RETRY_DELAY_SECONDS}s before next primary Gemini attempt for {mentioned_post_uri}...")
                time.sleep(GEMINI_RETRY_DELAY_SECONDS)
        # End of primary Gemini retry loop

        # If primary model failed to produce any text or an image prompt, skip.
        if not gemini_response_text and not image_prompt_for_imagen and not video_prompt:
            logging.error(f"All {MAX_GEMINI_RETRIES} attempts to get content from primary Gemini failed for {mentioned_post_uri}. Skipping reply.")
            return

        # --- Media Generation, if requested by primary model ---
        media_data_bytes = None
        media_type = None
        generated_alt_text = ""
        content_policy_message = None

        if video_prompt:
            video_result = generate_video_with_veo2(video_prompt, genai_client)
            if isinstance(video_result, bytes):
                # Successful video generation
                media_data_bytes = video_result
                media_type = 'video'
                generated_alt_text = clean_alt_text(video_prompt)
                logging.info(f"Successfully generated video. Size: {len(media_data_bytes)} bytes")
            elif isinstance(video_result, str):
                # Content policy violation - use the helpful message
                content_policy_message = video_result
                logging.info("Video generation failed due to content policy. Using explanatory message.")
            else:
                # Technical failure (None)
                logging.error("Video generation failed due to technical issues.")
                
        elif image_prompt_for_imagen:
            image_result = generate_image_with_imagen3(image_prompt_for_imagen, genai_client)
            if isinstance(image_result, bytes):
                # Successful image generation
                media_data_bytes = image_result
                media_type = 'image'
                generated_alt_text = clean_alt_text(image_prompt_for_imagen)
                logging.info(f"Successfully generated image. Size: {len(media_data_bytes)} bytes")
            elif isinstance(image_result, str):
                # Content policy violation - use the helpful message
                content_policy_message = image_result
                logging.info("Image generation failed due to content policy. Using explanatory message.")
            else:
                # Technical failure (None)
                logging.error("Image generation failed due to technical issues.")
        
        # Handle different failure scenarios
        if (video_prompt or image_prompt_for_imagen) and not media_data_bytes:
            if content_policy_message:
                # Content policy failure - use the helpful message instead of generic fallback
                if gemini_response_text:
                    gemini_response_text += f"\n\n{content_policy_message}"
                else:
                    gemini_response_text = content_policy_message
                logging.info("Using content policy explanation message.")
            else:
                # Technical failure - use generic fallback
                if gemini_response_text:
                    gemini_response_text += "\n(Sorry, I tried to generate something for you, but it didn't work out this time!)"
                    logging.info("Media generation failed technically, but text response is available. Appending generic note.")
                else:
                    # If only media was requested but it failed technically, and no other text was provided, don't post.
                    logging.warning(f"Primary model requested media but generation failed technically, and no fallback text from primary. Skipping reply.")
                    return

        # If no text at all (e.g. primary model only outputted a media prompt and generation failed), skip.
        if not gemini_response_text and not media_data_bytes:
            logging.warning(f"Neither text nor media could be generated for {mentioned_post_uri}. Skipping.")
            return

        # --- Post Splitting and Formatting ---
        # Combine text and grounding info before splitting
        final_response_text = gemini_response_text.strip() if gemini_response_text else ""
        
        # Check for grounding attribution to create separate post
        was_grounded = False
        search_queries = []
        # The response object has candidates, we'll check the first one.
        if primary_gemini_response_obj.candidates and hasattr(primary_gemini_response_obj.candidates[0], 'grounding_metadata'):
             grounding_metadata = primary_gemini_response_obj.candidates[0].grounding_metadata
             # Check if the metadata is not empty or null
             if grounding_metadata:
                 was_grounded = True
                 # Extract web search queries for Google Search Suggestions
                 if hasattr(grounding_metadata, 'web_search_queries') and grounding_metadata.web_search_queries:
                     search_queries = grounding_metadata.web_search_queries
                     logging.info(f"Found {len(search_queries)} search queries for Google Search Suggestions")

        # Split the final text into multiple posts if necessary
        post_texts = split_text_for_bluesky(final_response_text)
        if not post_texts: # If splitting results in no text, but we have media, create one empty post
            if media_data_bytes:
                post_texts = [""]
            else: # No text and no media, so nothing to post
                logging.warning("Final response text and media are both empty. Skipping.")
                return

        # --- Determine Initial Root and Parent for the Reply Thread --- 
        initial_parent_ref = at_models.ComAtprotoRepoStrongRef.Main(cid=target_post.cid, uri=target_post.uri)
        initial_root_ref = None
        post_record = target_post.record
        if isinstance(post_record, at_models.AppBskyFeedPost.Record) and post_record.reply:
            initial_root_ref = post_record.reply.root
        if initial_root_ref is None:
            initial_root_ref = initial_parent_ref
        
        # --- Send Reply Thread ---
        current_parent_ref = initial_parent_ref
        current_root_ref = initial_root_ref
        
        for i, post_text in enumerate(post_texts[:MAX_REPLY_THREAD_DEPTH]):
            embed_to_post = None
            # Only attach media to the very first post of the thread
            if i == 0 and media_data_bytes is not None and bsky_client:
                try:
                    # Skip extremely small files that are likely invalid
                    if len(media_data_bytes) < 1000:
                        logging.warning(f"Media data too small to be valid ({len(media_data_bytes)} bytes). Skipping upload.")
                    
                    elif media_type == 'image':
                        logging.info(f"Uploading generated image to Bluesky... Original Size: {len(media_data_bytes)} bytes")
                        # Make sure we're working with raw bytes for upload_blob
                        # If it's base64 string, decode it first
                        if isinstance(media_data_bytes, str):
                            media_data_bytes = base64.b64decode(media_data_bytes)
                        compressed_image = compress_image(media_data_bytes)
                        response = bsky_client.com.atproto.repo.upload_blob(compressed_image)
                        if response and response.blob:
                            logging.info(f"Image uploaded. CID: {response.blob.cid}")
                            if len(generated_alt_text) > 300:
                                generated_alt_text = generated_alt_text[:297] + "..."
                            image_for_embed = at_models.AppBskyEmbedImages.Image(alt=generated_alt_text, image=response.blob)
                            embed_to_post = at_models.AppBskyEmbedImages.Main(images=[image_for_embed])
                        else:
                            logging.error("Failed to upload image.")
    
                    elif media_type == 'video':
                        logging.info(f"Uploading generated video to Bluesky... Size: {len(media_data_bytes)} bytes")
                        response = bsky_client.com.atproto.repo.upload_blob(media_data_bytes)
                        if response and response.blob:
                            logging.info(f"Video uploaded. CID: {response.blob.cid}")
                            embed_to_post = at_models.AppBskyEmbedVideo.Main(video=response.blob, alt=generated_alt_text)
                        else:
                            logging.error("Failed to upload video.")
    
                except Exception as e:
                    logging.error(f"Error uploading media to Bluesky: {e}", exc_info=True)

            # --- Facet Generation (for each post) ---
            facets = generate_facets_for_text(post_text, bsky_client)

            # Send the reply post
            logging.info(f"📤 Sending reply {i+1}/{len(post_texts)}")
            facets_to_send = facets if facets else None
    
            try:
                # Apply rate limiting for Bluesky API
                rate_limiter.wait_if_needed_bluesky()
                
                response = bsky_client.send_post(
                    text=post_text,
                    reply_to=at_models.AppBskyFeedPost.ReplyRef(root=current_root_ref, parent=current_parent_ref),
                    embed=embed_to_post,
                    facets=facets_to_send
                )
                logging.info(f"✅ Sent reply {i+1}")
                
                # Update the parent reference for the *next* post in the thread
                current_parent_ref = at_models.ComAtprotoRepoStrongRef.Main(cid=response.cid, uri=response.uri)
    
            except AtProtocolError as api_error:
                logging.error(f"Bluesky API error creating post {i+1}: {api_error}", exc_info=True)
                # If one post in the thread fails, stop trying to post the rest
                break 
            except Exception as post_error:
                logging.error(f"Error creating post {i+1}: {post_error}", exc_info=True)
                break
        
        # --- Send Grounding Information as Separate Post ---
        if was_grounded and search_queries:
            try:
                # Remove duplicates while preserving order
                seen = set()
                unique_queries = []
                for query in search_queries:
                    if query.lower() not in seen:
                        seen.add(query.lower())
                        unique_queries.append(query)
                
                if unique_queries:
                    # Try with 3 suggestions first, then reduce if too long
                    for max_suggestions in [3, 2, 1]:
                        queries_to_use = unique_queries[:max_suggestions]
                        search_sections = []
                        
                        for query in queries_to_use:
                            # URL encode the search query
                            encoded_query = urllib.parse.quote_plus(query)
                            search_url = f"https://www.google.com/search?q={encoded_query}"
                            # Format each suggestion as query on one line, link on next line
                            search_sections.append(f"🔍 {query}:\n{search_url}")
                        
                        # Format the grounding post with dashed header and line breaks for readability
                        grounding_post_text = "-- Grounded with Google Search --\n\n" + "\n\n".join(search_sections)
                        
                        # Check if this fits within the 300 character limit
                        if len(grounding_post_text) <= 300:
                            logging.info(f"Creating grounding post with {len(queries_to_use)} Google Search suggestions (fits in {len(grounding_post_text)} characters).")
                            
                            # Create facets for mentions and links in the grounding post
                            grounding_facets = generate_facets_for_text(grounding_post_text, bsky_client)
                            grounding_facets_to_send = grounding_facets if grounding_facets else None
                            
                            # Send the single grounding post
                            response = bsky_client.send_post(
                                text=grounding_post_text,
                                reply_to=at_models.AppBskyFeedPost.ReplyRef(root=current_root_ref, parent=current_parent_ref),
                                facets=grounding_facets_to_send
                            )
                            logging.info(f"Successfully sent grounding post. URI: {response.uri}")
                            break
                        else:
                            logging.info(f"Grounding post with {len(queries_to_use)} suggestions is too long ({len(grounding_post_text)} characters). Trying with fewer suggestions.")
                
            except Exception as grounding_error:
                logging.error(f"Error creating grounding post: {grounding_error}", exc_info=True)
        
    except AtProtocolError as e:
        logging.error(f"Bluesky API error processing mention {mentioned_post_uri}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error processing mention {mentioned_post_uri}: {e}", exc_info=True)
    finally:
        # Clean up downloaded media data to free memory
        try:
            if 'image_parts' in locals():
                del image_parts
            if 'video_parts' in locals():
                del video_parts
            if 'downloaded_media_data' in locals():
                del downloaded_media_data
            gc.collect()
            logging.debug("Memory cleanup performed after processing mention")
        except Exception as cleanup_error:
            logging.warning(f"Error during memory cleanup: {cleanup_error}")

def generate_video_with_veo2(prompt: str, client: genai.Client) -> bytes | str | None:
    """
    Generates a video using Veo 2 and returns the video bytes or error message.
    Returns:
        bytes: Video data if successful
        str: User-friendly error message if content policy violation
        None: Technical failure (will show generic fallback)
    """
    logging.info(f"Generating video with Veo 2 for prompt: '{prompt}'")
    
    for attempt in range(MAX_VIDEO_GENERATION_RETRIES):
        try:
            logging.info(f"🎬 Video generation attempt {attempt + 1}/{MAX_VIDEO_GENERATION_RETRIES}")
            
            video_config = GenerateVideosConfig(
                number_of_videos=1,
                fps=24,
                duration_seconds=8,
                # person_generation=VIDEO_PERSON_GENERATION,  # Removing this parameter as it may not be supported
            )

            operation = client.models.generate_videos(
                model=VEO_MODEL_NAME,
                prompt=prompt,
                config=video_config,
            )

            logging.info(f"Video generation started (attempt {attempt + 1}). Polling for completion...")
            
            # Polling for completion, with a timeout
            POLL_INTERVAL_SECONDS = 15
            MAX_POLLING_ATTEMPTS = 40 # 15s * 40 = 600s = 10 minutes timeout
            for _ in range(MAX_POLLING_ATTEMPTS):
                if operation.done:
                    break
                logging.info(f"Video not ready. Checking again in {POLL_INTERVAL_SECONDS} seconds...")
                time.sleep(POLL_INTERVAL_SECONDS)
                operation = client.operations.get(operation)

            if not operation.done:
                error_msg = f"Video generation timed out after 10 minutes for prompt: '{prompt}' (attempt {attempt + 1})"
                logging.error(error_msg)
                if attempt == MAX_VIDEO_GENERATION_RETRIES - 1:
                    # Only send DM on final attempt failure for timeouts (technical issue)
                    send_developer_dm(error_msg, "VIDEO GENERATION TIMEOUT", allow_public_fallback=False)
                    return None
                else:
                    # Wait before retrying timeouts
                    logging.info(f"Waiting {VIDEO_RETRY_DELAY_SECONDS}s before retry...")
                    time.sleep(VIDEO_RETRY_DELAY_SECONDS)
                    continue

            result = operation.result
            logging.info(f"Video generation operation result (attempt {attempt + 1}): {result}")
            if hasattr(result, '__dict__'):
                logging.info(f"Result attributes: {result.__dict__}")
            
            if not result or not result.generated_videos:
                debug_info = f"Result exists: {result is not None}"
                if result:
                    debug_info += f", has generated_videos attr: {hasattr(result, 'generated_videos')}"
                    if hasattr(result, 'generated_videos'):
                        debug_info += f", generated_videos value: {result.generated_videos}"
                
                error_msg = f"Video generation failed for prompt: '{prompt}' (attempt {attempt + 1}). API returned no videos. Debug: {debug_info}"
                logging.error(error_msg)
                
                # Check if this looks like a content policy failure
                if is_content_policy_failure(error_msg, result, prompt):
                    logging.info(f"Video generation failure appears to be content policy related. Returning user message.")
                    return get_content_policy_message("video", prompt)
                
                # Technical failure - retry if attempts remain
                if attempt == MAX_VIDEO_GENERATION_RETRIES - 1:
                    # Only send DM on final attempt failure for technical issues
                    send_developer_dm(error_msg, "VIDEO GENERATION FAILURE", allow_public_fallback=False)
                    return None
                else:
                    # Wait before retrying technical failures
                    logging.info(f"Waiting {VIDEO_RETRY_DELAY_SECONDS}s before retry...")
                    time.sleep(VIDEO_RETRY_DELAY_SECONDS)
                    continue

            generated_video = result.generated_videos[0]
            logging.info(f"Video generated successfully on attempt {attempt + 1}")

            # Download the video content
            video_bytes = client.files.download(file=generated_video.video)

            logging.info(f"Successfully downloaded video. Size: {len(video_bytes)} bytes")
            return video_bytes

        except Exception as e:
            error_msg = f"Veo 2 video generation failed with exception for prompt '{prompt}' (attempt {attempt + 1}): {e}"
            logging.error(error_msg, exc_info=True)
            
            # Check if this looks like a content policy failure
            if is_content_policy_failure(str(e), None, prompt):
                logging.info(f"Video generation exception appears to be content policy related. Returning user message.")
                return get_content_policy_message("video", prompt)
            
            # Technical failure - retry if attempts remain
            if attempt == MAX_VIDEO_GENERATION_RETRIES - 1:
                # Only send DM on final attempt failure for technical issues
                send_developer_dm(error_msg, "VIDEO GENERATION ERROR", allow_public_fallback=False)
                return None
            else:
                # Wait before retrying technical failures
                logging.info(f"Waiting {VIDEO_RETRY_DELAY_SECONDS}s before retry...")
                time.sleep(VIDEO_RETRY_DELAY_SECONDS)
    
    return None

def generate_image_with_imagen3(prompt: str, client: genai.Client) -> bytes | str | None:
    """
    Generates an image using Imagen 3 and returns the image bytes or error message.
    Returns:
        bytes: Image data if successful
        str: User-friendly error message if content policy violation
        None: Technical failure (will show generic fallback)
    """
    logging.info(f"Generating image with Imagen 3 for prompt: '{prompt}'")
    
    for attempt in range(MAX_IMAGE_GENERATION_RETRIES):
        try:
            logging.info(f"🎨 Image generation attempt {attempt + 1}/{MAX_IMAGE_GENERATION_RETRIES}")
            
            result = client.models.generate_images(
                model=f"models/{IMAGEN_MODEL_NAME}",
                prompt=prompt,
                config={
                    "number_of_images": 1,
                    "output_mime_type": "image/jpeg",
                    "person_generation": IMAGE_PERSON_GENERATION,
                    "aspect_ratio": "1:1",
                },
            )

            if not result.generated_images:
                error_msg = f"Image generation failed for prompt: '{prompt}' (attempt {attempt + 1}). No images generated by Imagen 3."
                logging.warning(error_msg)
                
                # Check if this looks like a content policy failure
                if is_content_policy_failure(error_msg, result, prompt):
                    logging.info(f"Image generation failure appears to be content policy related. Returning user message.")
                    return get_content_policy_message("image", prompt)
                
                # Technical failure - retry if attempts remain
                if attempt == MAX_IMAGE_GENERATION_RETRIES - 1:
                    # Only send DM on final attempt failure for technical issues
                    send_developer_dm(error_msg, "IMAGE GENERATION FAILURE", allow_public_fallback=False)
                    return None
                else:
                    # Wait before retrying technical failures
                    logging.info(f"Waiting {IMAGE_RETRY_DELAY_SECONDS}s before retry...")
                    time.sleep(IMAGE_RETRY_DELAY_SECONDS)
                    continue

            # Assuming we only care about the first image if multiple are returned
            generated_image = result.generated_images[0]
            if hasattr(generated_image, 'image') and hasattr(generated_image.image, 'image_bytes'):
                image_bytes = generated_image.image.image_bytes
                logging.info(f"Successfully generated image on attempt {attempt + 1}. Size: {len(image_bytes)} bytes")
                return image_bytes
            else:
                error_msg = f"Image generation failed for prompt: '{prompt}' (attempt {attempt + 1}). Generated image object does not have expected structure."
                logging.error(error_msg)
                
                # Structure errors are typically technical, not policy
                if attempt == MAX_IMAGE_GENERATION_RETRIES - 1:
                    # Only send DM on final attempt failure for technical issues
                    send_developer_dm(error_msg, "IMAGE GENERATION STRUCTURE ERROR", allow_public_fallback=False)
                    return None
                else:
                    # Wait before retrying technical failures
                    logging.info(f"Waiting {IMAGE_RETRY_DELAY_SECONDS}s before retry...")
                    time.sleep(IMAGE_RETRY_DELAY_SECONDS)
                    continue

        except Exception as e:
            error_msg = f"Imagen 3 image generation failed with exception for prompt '{prompt}' (attempt {attempt + 1}): {e}"
            logging.error(error_msg, exc_info=True)
            
            # Check if this looks like a content policy failure
            if is_content_policy_failure(str(e), None, prompt):
                logging.info(f"Image generation exception appears to be content policy related. Returning user message.")
                return get_content_policy_message("image", prompt)
            
            # Technical failure - retry if attempts remain
            if attempt == MAX_IMAGE_GENERATION_RETRIES - 1:
                # Only send DM on final attempt failure for technical issues
                send_developer_dm(error_msg, "IMAGE GENERATION ERROR", allow_public_fallback=False)
                return None
            else:
                # Wait before retrying technical failures
                logging.info(f"Waiting {IMAGE_RETRY_DELAY_SECONDS}s before retry...")
                time.sleep(IMAGE_RETRY_DELAY_SECONDS)
    
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

def download_video_from_url(url: str, max_size_mb: float = 20.0, timeout: int = 30) -> bytes | None:
    """
    Downloads a video from a URL and returns the raw bytes.
    Returns None if the download fails.
    
    Args:
        url: The URL to download from
        max_size_mb: Maximum size of the video in MB
        timeout: Timeout in seconds for the request
    """
    try:
        logging.info(f"Downloading video from URL: {url}")
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code != 200:
            logging.error(f"Failed to download video from {url}. Status code: {response.status_code}")
            return None
            
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('video/'):
            logging.warning(f"URL does not contain a video. Content-Type: {content_type}")
            return None
        
        # Get content length if available
        content_length = response.headers.get('Content-Length')
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            logging.warning(f"Video too large ({int(content_length) / (1024 * 1024):.2f} MB). Skipping download.")
            return None
            
        # Download video with size monitoring
        video_bytes = BytesIO()
        total_size = 0
        max_size_bytes = max_size_mb * 1024 * 1024
        
        for chunk in response.iter_content(chunk_size=8192):
            total_size += len(chunk)
            if total_size > max_size_bytes:
                logging.warning(f"Video download exceeded max size of {max_size_mb} MB. Aborting.")
                return None
            video_bytes.write(chunk)
        
        final_bytes = video_bytes.getvalue()
        logging.info(f"Successfully downloaded video. Size: {len(final_bytes) / 1024:.2f} KB")
        return final_bytes
    except requests.exceptions.Timeout:
        logging.error(f"Timeout downloading video from {url} after {timeout} seconds")
        return None
    except Exception as e:
        logging.error(f"Error downloading video from {url}: {e}")
        return None

def download_video_blob_from_pds(blob_cid: str, pds_endpoint: str, author_did: str, timeout: int = 30) -> bytes | None:
    """
    Downloads a video blob from a Bluesky PDS using the getBlob endpoint.
    Returns None if the download fails.
    
    Args:
        blob_cid: The blob CID to download
        pds_endpoint: The PDS endpoint URL
        author_did: The DID of the post author
        timeout: Timeout in seconds for the request
    """
    try:
        logging.info(f"Downloading video blob {blob_cid} from PDS: {pds_endpoint}")
        
        # Construct the getBlob URL
        blob_url = f"{pds_endpoint}/xrpc/com.atproto.sync.getBlob?did={author_did}&cid={blob_cid}"
        
        response = requests.get(blob_url, timeout=timeout, stream=True)
        if response.status_code != 200:
            logging.error(f"Failed to download video blob {blob_cid}. Status code: {response.status_code}")
            return None
            
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('video/'):
            logging.warning(f"Blob {blob_cid} does not contain a video. Content-Type: {content_type}")
            return None
        
        # Get content length if available
        content_length = response.headers.get('Content-Length')
        max_size_bytes = 20 * 1024 * 1024  # 20MB limit
        if content_length and int(content_length) > max_size_bytes:
            logging.warning(f"Video blob {blob_cid} too large ({int(content_length) / (1024 * 1024):.2f} MB). Skipping download.")
            return None
            
        # Download video with size monitoring
        video_bytes = BytesIO()
        total_size = 0
        
        for chunk in response.iter_content(chunk_size=8192):
            total_size += len(chunk)
            if total_size > max_size_bytes:
                logging.warning(f"Video blob {blob_cid} download exceeded max size of 20MB. Aborting.")
                return None
            video_bytes.write(chunk)
        
        final_bytes = video_bytes.getvalue()
        logging.info(f"Successfully downloaded video blob {blob_cid}. Size: {len(final_bytes) / 1024:.2f} KB")
        return final_bytes
    except requests.exceptions.Timeout:
        logging.error(f"Timeout downloading video blob {blob_cid} after {timeout} seconds")
        return None
    except Exception as e:
        logging.error(f"Error downloading video blob {blob_cid}: {e}")
        return None

def is_mention_of_bot(post_text: str) -> bool:
    """Check if a post text contains a mention of the bot."""
    if not post_text:
        return False
    
    # Look for @handle mentions
    handle_patterns = [
        f"@{BLUESKY_HANDLE}",
        f"@{BLUESKY_HANDLE.lower()}",
        f"@{BLUESKY_HANDLE.upper()}"
    ]
    
    for pattern in handle_patterns:
        if pattern in post_text:
            return True
    
    return False

def should_process_jetstream_event(event: dict) -> bool:
    """Determine if a Jetstream event should be processed by the bot."""
    try:
        # Only process commit events
        if event.get("kind") != "commit":
            return False
        
        commit = event.get("commit", {})
        
        # Only process create operations (new posts)
        if commit.get("operation") != "create":
            return False
        
        # Only process posts (not likes, follows, etc.)
        if commit.get("collection") != "app.bsky.feed.post":
            return False
        
        # Don't process posts from the bot itself
        event_did = event.get("did")
        if event_did == bot_did:
            return False
        
        record = commit.get("record", {})
        post_text = record.get("text", "")
        
        # Check if this is a mention of the bot
        if is_mention_of_bot(post_text):
            logging.debug(f"Found mention in post: {post_text[:100]}...")
            return True
        
        # Check if this is a reply to the bot
        reply_info = record.get("reply")
        if reply_info:
            parent_uri = reply_info.get("parent", {}).get("uri", "")
            root_uri = reply_info.get("root", {}).get("uri", "")
            
            # Check if the immediate parent contains the bot's DID (direct reply to bot)
            if bot_did and bot_did in parent_uri:
                logging.debug(f"Found direct reply to bot post")
                return True
            
            # Don't process replies to other users in a thread, even if bot is in root
            # This prevents the bot from replying to user-to-user conversations
        
        return False
        
    except Exception as e:
        logging.error(f"Error checking if event should be processed: {e}")
        return False

async def connect_to_jetstream():
    """Connect to Jetstream and yield events."""
    while True:
        try:
            # Filter to only receive posts
            params = {
                "wantedCollections": ["app.bsky.feed.post"]
            }
            uri = f"{JETSTREAM_ENDPOINT}?" + urllib.parse.urlencode(params, doseq=True)
            
            logging.info(f"Connecting to Jetstream: {uri}")
            
            async with websockets.connect(uri) as websocket:
                logging.info("✅ Connected to Jetstream")
                
                async for message in websocket:
                    try:
                        event = json.loads(message)
                        if should_process_jetstream_event(event):
                            yield event
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse Jetstream message: {e}")
                    except Exception as e:
                        logging.error(f"Error processing Jetstream message: {e}")
                        
        except websockets.exceptions.ConnectionClosed:
            logging.warning(f"Jetstream connection closed. Reconnecting in {JETSTREAM_RECONNECT_DELAY}s...")
            await asyncio.sleep(JETSTREAM_RECONNECT_DELAY)
        except Exception as e:
            error_msg = f"Jetstream connection error: {e}. Reconnecting in {JETSTREAM_RECONNECT_DELAY}s..."
            logging.error(error_msg)
            # Send DM for persistent connection issues (only if we've been disconnected for a while)
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                send_developer_dm(f"Jetstream connection issues: {str(e)}", "CONNECTION WARNING", allow_public_fallback=False)
            await asyncio.sleep(JETSTREAM_RECONNECT_DELAY)

def process_jetstream_event(event: dict, genai_client_ref: genai.Client):
    """Process a single Jetstream event that represents a mention or reply."""
    global bsky_client, processed_uris_this_run
    if not bsky_client:
        logging.error("Bluesky client not initialized. Cannot process event.")
        return

    try:
        # Construct the AT URI for the post
        did = event.get("did")
        commit = event.get("commit", {})
        collection = commit.get("collection")
        rkey = commit.get("rkey")
        
        if not all([did, collection, rkey]):
            logging.error(f"Missing required fields in event: {event}")
            return
            
        post_uri = f"at://{did}/{collection}/{rkey}"
        
        # Thread-safe marking as seen for this run before processing
        with _processed_uris_lock:
            processed_uris_this_run[post_uri] = None
            if len(processed_uris_this_run) > MAX_PROCESSED_URIS_CACHE:
                processed_uris_this_run.popitem(last=False)
        
        logging.info(f"🔄 Processing Jetstream event for post: {post_uri}")
        
        # Get the full thread context for this post
        params = GetPostThreadParams(uri=post_uri, depth=MAX_THREAD_DEPTH_FOR_CONTEXT)
        thread_view_response = bsky_client.app.bsky.feed.get_post_thread(params=params)
        
        if not isinstance(thread_view_response.thread, at_models.AppBskyFeedDefs.ThreadViewPost):
            logging.warning(f"Could not fetch thread or thread is not a ThreadViewPost for {post_uri}")
            return

        thread_view_of_mentioned_post = thread_view_response.thread
        target_post = thread_view_of_mentioned_post.post
        if not target_post:
            logging.warning(f"Thread view for {post_uri} does not contain a post.")
            return

        # Check thread depth limits
        thread_length = get_thread_length(thread_view_of_mentioned_post)
        logging.debug(f"Current conversation thread length is {thread_length}.")
        if thread_length >= MAX_CONVERSATION_THREAD_DEPTH:
            logging.info(f"🔁 Thread too long ({thread_length}), sending limit message.")
            try:
                canned_parent_ref = at_models.ComAtprotoRepoStrongRef.Main(cid=target_post.cid, uri=target_post.uri)
                canned_root_ref = None
                post_record = target_post.record
                if isinstance(post_record, at_models.AppBskyFeedPost.Record) and post_record.reply:
                    canned_root_ref = post_record.reply.root
                if canned_root_ref is None:
                    canned_root_ref = canned_parent_ref
                
                bsky_client.send_post(
                    text="Sorry, this thread has gotten very long! If you'd like to continue our conversation, please start a new thread.",
                    reply_to=at_models.AppBskyFeedPost.ReplyRef(root=canned_root_ref, parent=canned_parent_ref)
                )
                logging.info("✅ Sent thread depth limit message.")
            except Exception as e:
                logging.error(f"Failed to send thread depth limit message: {e}")
            return

        # Check for duplicate replies (to prevent multiple responses to the same post)
        if thread_view_of_mentioned_post.replies:
            for reply_in_thread in thread_view_of_mentioned_post.replies:
                if (reply_in_thread.post and reply_in_thread.post.author and 
                    reply_in_thread.post.author.handle == BLUESKY_HANDLE):
                    logging.debug(f"Found pre-existing bot reply to {post_uri}. Skipping.")
                    return

        # Validate required fields before processing
        if not target_post.author or not target_post.author.handle or not target_post.author.did:
            logging.error(f"Post {post_uri} missing required author information")
            return

        # Create a mock notification object to reuse existing logic with proper validation
        class MockAuthor:
            def __init__(self, handle: str, did: str):
                self.handle = handle
                self.did = did

        class MockNotification:
            def __init__(self, uri: str, author: MockAuthor, reason: str):
                self.uri = uri
                self.author = author
                self.reason = reason

        post_text = commit.get("record", {}).get("text", "")
        reason = 'mention' if is_mention_of_bot(post_text) else 'reply'
        
        mock_notification = MockNotification(
            uri=post_uri,
            author=MockAuthor(target_post.author.handle, target_post.author.did),
            reason=reason
        )
        
        # Use existing process_mention logic
        process_mention(mock_notification, genai_client_ref)
        
    except Exception as e:
        logging.error(f"Error processing Jetstream event: {e}", exc_info=True)

async def jetstream_listener():
    """Main Jetstream listener loop."""
    global genai_client
    if not genai_client:
        logging.error("GenAI client not initialized. Cannot start Jetstream listener.")
        return
    
    logging.info("🚀 Starting Jetstream listener...")
    
    async for event in connect_to_jetstream():
        try:
            # Run the processing in a thread to avoid blocking the async loop
            threading.Thread(
                target=process_jetstream_event,
                args=(event, genai_client),
                daemon=True
            ).start()
        except Exception as e:
            logging.error(f"Error starting thread for event processing: {e}")
            # Don't send DM for every thread error, just log it

def catch_up_missed_notifications(bsky_client_ref: Client, genai_client_ref: genai.Client):
    """Fetches and processes past notifications to catch up on missed interactions."""
    global processed_uris_this_run # Ensure access to the global set
    if not (bsky_client_ref and genai_client_ref):
        logging.error("Bluesky client or GenAI client not available for catch-up.")
        return

    logging.info(f"Starting catch-up: Fetching last {CATCH_UP_NOTIFICATION_LIMIT} notifications...")
    try:
        params = ListNotificationsParams(limit=CATCH_UP_NOTIFICATION_LIMIT)
        response = bsky_client_ref.app.bsky.notification.list_notifications(params=params)

        if response and response.notifications:
            logging.info(f"Catch-up: Fetched {len(response.notifications)} notifications.")
            # Sort by indexedAt to process older notifications first, though order for catch-up is less critical
            # as we are checking for already replied posts.
            sorted_notifications = sorted(response.notifications, key=lambda n: n.indexed_at)

            processed_in_catchup = 0
            for notification in sorted_notifications:
                # Skip if notification is from the bot itself
                if notification.author.handle == BLUESKY_HANDLE:
                    logging.debug(f"Catch-up: Skipping notification {notification.uri} from bot itself.")
                    continue
                
                # Skip if already processed in this current run (e.g. if catch-up is somehow run multiple times quickly)
                # This is less critical here than in main_loop but good for consistency.
                if notification.uri in processed_uris_this_run:
                    logging.debug(f"Catch-up: Skipping notification {notification.uri} already processed in this run/session.")
                    continue

                # We are re-analyzing, so we don't strictly skip based on `is_read` here,
                # as the bot might have gone down before processing or marking as read.
                # The `process_mention` function has internal duplicate reply prevention.

                logging.debug(f"Catch-up: Analyzing notification: type={notification.reason}, from={notification.author.handle}, uri={notification.uri}")
                if notification.reason in ['mention', 'reply']:
                    process_mention(notification, genai_client_ref)
                    processed_in_catchup += 1
                    logging.info(f"📬 Processed {notification.reason} from @{notification.author.handle}")
                else:
                    logging.debug(f"Catch-up: Skipping notification {notification.uri} with reason '{notification.reason}'.")
            logging.info(f"Catch-up: Finished processing {processed_in_catchup} relevant notifications from the fetched batch.")
        else:
            logging.info("Catch-up: No notifications found or error in fetching for catch-up.")
    except AtProtocolError as e:
        logging.error(f"Catch-up: Bluesky API error during catch-up: {e}")
    except Exception as e:
        logging.error(f"Catch-up: Unexpected error during catch-up: {e}", exc_info=True)
    logging.info("Catch-up process complete.")

async def main_bot_loop():
    """Main async loop for the bot using Jetstream."""
    global bsky_client, genai_client
    
    if not (bsky_client and genai_client):
        logging.critical("Bluesky client or GenAI services not initialized. Exiting main loop.")
        return
    
    logging.info("🚀 Bot starting main async loop with Jetstream...")
    
    try:
        # Start the Jetstream listener
        await jetstream_listener()
    except Exception as e:
        log_critical_error(f"Fatal error in main bot loop", e)

async def main():
    global bsky_client, genai_client # Declare intent to modify globals

    logging.info("🤖 Bot starting...")
    
    bsky_client = initialize_bluesky_client()
    genai_client = initialize_genai_services()

    if bsky_client and genai_client: # Check all requirements are met
        # Send startup notification to developer
        try:
            startup_msg = f"🤖 Bot @{BLUESKY_HANDLE} started successfully!\n\nFeatures enabled:\n- Jetstream real-time monitoring\n- Gemini AI responses\n- Image generation (Imagen 3)\n- Video generation (Veo 2)\n- Thread management\n- Rate limiting\n\nMemory: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB"
            send_startup_notification(startup_msg)
        except Exception as startup_error:
            logging.warning(f"Failed to send startup notification: {startup_error}")
        
        # Perform catch-up for missed notifications before starting the main loop
        logging.info("📬 Performing notification catch-up...")
        catch_up_missed_notifications(bsky_client, genai_client)
        
        # Start the main Jetstream loop
        logging.info("🌊 Starting Jetstream-based bot loop...")
        await main_bot_loop()
    else:
        if not bsky_client:
            log_critical_error("❌ Failed to initialize Bluesky client. Bot cannot start.")
        if not genai_client:
            log_critical_error("❌ Failed to initialize GenAI services. Bot cannot start.")

if __name__ == "__main__":
    asyncio.run(main())