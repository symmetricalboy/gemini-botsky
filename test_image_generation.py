import os
import json
import base64
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if the API key is available
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please add it to your .env file.")

# Define the model
MODEL_NAME = "gemini-2.0-flash-preview-image-generation"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

def generate_image(prompt):
    """Generate an image using Gemini and save it locally."""
    print(f"Generating image with prompt: {prompt}")
    
    try:
        # Create the request payload based on latest documentation
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
                "responseModalities": ["TEXT", "IMAGE"]
            }
        }
        
        # Make the API request
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        # Check if the request was successful
        if response.status_code != 200:
            print(f"API request failed with status code {response.status_code}")
            print(f"Error message: {response.text}")
            
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
            
            print("Trying alternative payload structure...")
            response = requests.post(
                API_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps(alternative_payload)
            )
            
            if response.status_code != 200:
                print(f"Alternative payload also failed with status code {response.status_code}")
                print(f"Error message: {response.text}")
                return False
        
        # Parse the response
        response_data = response.json()
        
        # Process the response to find image data
        image_saved = False
        text_response = ""
        
        if "candidates" in response_data and len(response_data["candidates"]) > 0:
            candidate = response_data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    # Check for text
                    if "text" in part:
                        text_response += part["text"]
                        print(f"Model response text: {part['text']}")
                    
                    # Check for image
                    if "inlineData" in part:
                        image_data = part["inlineData"]["data"]
                        image_bytes = base64.b64decode(image_data)
                        
                        # Save the image
                        image = Image.open(BytesIO(image_bytes))
                        filename = f"generated_image_{prompt[:20].replace(' ', '_').replace(':', '_')}.png"
                        image.save(filename)
                        print(f"Image saved as: {filename}")
                        image_saved = True
        
        if not image_saved:
            print("No image was generated in the response.")
            print(f"Response structure: {json.dumps(response_data, indent=2)}")
        
        return image_saved
    except Exception as e:
        print(f"Image generation failed: {e}")
        return False

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
    
    try:
        # Create the request payload based on latest documentation
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": full_prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"]
            }
        }
        
        # Make the API request
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        # Check if the request was successful
        if response.status_code != 200:
            print(f"API request failed with status code {response.status_code}")
            print(f"Error message: {response.text}")
            
            # Try alternative payload structure
            alternative_payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": full_prompt
                            }
                        ]
                    }
                ],
                "generation_config": {
                    "response_mime_type": "image/png"
                }
            }
            
            print("Trying alternative payload structure...")
            response = requests.post(
                API_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps(alternative_payload)
            )
            
            if response.status_code != 200:
                print(f"Alternative payload also failed with status code {response.status_code}")
                print(f"Error message: {response.text}")
                return
        
        # Parse the response
        response_data = response.json()
        
        # Process the response
        has_image = False
        text_response = ""
        
        if "candidates" in response_data and len(response_data["candidates"]) > 0:
            candidate = response_data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    # Check for text
                    if "text" in part:
                        text_response += part["text"]
                    
                    # Check for image
                    if "inlineData" in part:
                        has_image = True
                        image_data = part["inlineData"]["data"]
                        image_bytes = base64.b64decode(image_data)
                        
                        # Save the image
                        image = Image.open(BytesIO(image_bytes))
                        filename = "generated_image_from_text_prompt.png"
                        image.save(filename)
                        print(f"Direct image generated and saved as: {filename}")
        
        # Process the text response
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
    except Exception as e:
        print(f"Text with image extraction failed: {e}")

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