# Bluesky Gemini Bot

This bot will reply to mentions on Bluesky using Google Gemini.

## Features

- Responds to mentions and replies on Bluesky
- Uses Google Gemini 2.0 for AI-powered responses
- Can generate images using Gemini's image generation capabilities
- Now with improved image understanding - can analyze images in posts!

## Setup

1. Clone this repository
2. Create a `.env` file with your credentials (see `.env.example`)
3. Install dependencies: `pip install -r requirements.txt`
4. Run the bot: `python bot.py`

## Image Processing

The bot can now analyze images in posts by:
1. Extracting image URLs from Bluesky posts
2. Downloading the images and converting them to base64
3. Sending the images to Gemini along with the text context
4. Generating a response based on both the text and image content

This enables the bot to respond to questions like "describe this image" or "what's in this photo?"

## Deployment

This bot is intended for deployment on Railway. 