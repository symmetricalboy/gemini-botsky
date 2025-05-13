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

# Define the model
MODEL_NAME = "gemini-2.0-flash-preview-image-generation"

def generate_image(prompt):
    """Generate an image using Gemini and save it locally."""
    print(f"Generating image with prompt: {prompt}")
    
    # Generate content with both text and image modalities
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=['IMAGE', 'TEXT']
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
            filename = f"generated_image_{prompt[:20].replace(' ', '_').replace(':', '_')}.png"
            image.save(filename)
            print(f"Image saved as: {filename}")
            image_saved = True
            
    if not image_saved:
        print("No image was generated in the response.")
    
    return image_saved

def test_text_with_image_extraction():
    """Test extracting image prompts from text responses."""
    print("\nTesting text response with IMAGE_PROMPT extraction...")
    
    # For models that don't support system instructions, include essential instructions in the prompt
    system_instructions = (
        "You are a helpful assistant that can generate both text and images. "
        "If the user requests or would benefit from an image, provide a text response "
        "followed by an image prompt on a new line starting with 'IMAGE_PROMPT: '. "
        "Keep your text response under an appropriate length (around 100-300 characters)."
    )
    
    user_prompt = "Show me a picture of a sunset over mountains"
    full_prompt = f"{system_instructions}\n\nUser request: {user_prompt}"
    
    print(f"Sending prompt: {user_prompt}")
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            response_modalities=['IMAGE', 'TEXT']
        )
    )
    
    # Process the response
    has_image = False
    text_response = ""
    
    # First check if there's an image directly in the response
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            text_response += part.text
        elif part.inline_data is not None:
            has_image = True
            # Save the image
            image = Image.open(BytesIO(part.inline_data.data))
            filename = "generated_image_from_text_prompt.png"
            image.save(filename)
            print(f"Direct image generated and saved as: {filename}")
    
    if text_response:
        print(f"Text response: {text_response[:100]}...")
        
        # Check for IMAGE_PROMPT in the text response
        if "IMAGE_PROMPT:" in text_response and not has_image:
            parts = text_response.split("IMAGE_PROMPT:", 1)
            text_only = parts[0].strip()
            image_prompt = parts[1].strip()
            
            print(f"Extracted text: {text_only}")
            print(f"Extracted image prompt: {image_prompt}")
            
            # Generate the image based on the extracted prompt
            generate_image(image_prompt)
        elif not has_image:
            print("No IMAGE_PROMPT found in the response.")
    else:
        print("No text response received.")

if __name__ == "__main__":
    print("=== GEMINI IMAGE GENERATION TEST ===")
    print(f"Using model: {MODEL_NAME}")
    
    choice = input("Choose test type:\n1. Direct image generation\n2. Text with image prompt extraction\nYour choice (1/2): ")
    
    if choice == "1":
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
    
    elif choice == "2":
        test_text_with_image_extraction()
    
    else:
        print("Invalid choice. Please run the script again and select 1 or 2.") 