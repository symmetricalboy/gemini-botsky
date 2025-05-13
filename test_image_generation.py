import os
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if the API key is available
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please add it to your .env file.")

# Initialize the Google Generative AI client
client = genai.Client(api_key=GEMINI_API_KEY)

def generate_image(prompt):
    """Generate an image using Gemini and save it locally."""
    print(f"Generating image with prompt: {prompt}")
    
    # Generate content with both text and image modalities
    response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']
        )
    )
    
    # Process the response
    image_saved = False
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(f"Model response text: {part.text}")
        elif part.inline_data is not None:
            # Save the image
            image = Image.open(BytesIO(part.inline_data.data))
            filename = f"generated_image_{prompt[:20].replace(' ', '_')}.png"
            image.save(filename)
            print(f"Image saved as: {filename}")
            image_saved = True
            
    if not image_saved:
        print("No image was generated in the response.")
    
    return image_saved

if __name__ == "__main__":
    # Get user input for the image prompt or use a default prompt
    user_prompt = input("Enter an image prompt (or press Enter for default): ")
    if not user_prompt:
        user_prompt = "A 3D rendered image of a happy robot playing with a cat in a futuristic setting"
    
    # Generate the image
    success = generate_image(user_prompt)
    
    if success:
        print("Image generation complete!")
    else:
        print("Failed to generate image.") 