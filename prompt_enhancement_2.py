import requests
import random
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Your Gemini API Key
API_KEY = "AIzaSyD96PJQN2yCDlHkdnDn4xG1Hb85JMRaJrQ"

# Configure the Gemini API with your API key
genai.configure(api_key=API_KEY)

# Initialize the sentence transformer model for semantic similarity
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to clean up the generated prompt output
def clean_output(text):
    # Split the response by newlines and process each line
    slogans = []
    for line in text.split('\n'):
        # Remove numbers, asterisks, and clean up the text
        cleaned_line = line.strip()
        cleaned_line = ''.join([char for char in cleaned_line if not char.isdigit()])
        cleaned_line = cleaned_line.replace(":", "").replace("*", "").replace("-", "").replace(")", "").strip()

        # Remove any unwanted words or lines, like 'Specific details' or headings
        if len(cleaned_line) > 5 and "Specific details" not in cleaned_line:
            slogans.append(cleaned_line)
    
    # Ensure we have exactly 10 slogans
    slogans = slogans[:10]
    return slogans

# Function to generate enhanced prompts using the Gemini API
def generate_enhanced_prompt(base_prompt, category="general"):
    generation_config = {
        "temperature": 0.7,  # Balance creativity with coherence
        "max_output_tokens": 200,  # Limit output to keep prompt concise
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    )

    # Create a prompt for the Gemini API to generate a more detailed version
    prompt = f"""Enhance the following prompt for creating an attractive poster image in Stable Diffusion. Add specific details about colors, lighting, textures, background elements, and style to improve image quality with minimal GPU usage.
    
    Base Prompt: '{base_prompt}'
    
    Enhanced Prompt:"""

    # Generate the enhanced prompt
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)
    return clean_output(response.text.strip())  # Clean and return the response

# Main function to generate and evaluate enhanced prompt
def generate_enhanced_prompt_final(user_input):
    # Generate an enhanced prompt using Gemini API
    enhanced_prompt = generate_enhanced_prompt(user_input)
    
    # Prepare the output data as a dictionary
    output_data = {
        "base_prompt": user_input,
        "enhanced_prompt": enhanced_prompt,
    }
    
    return output_data

# Example usage
user_input = "A technical hackathon with exciting cash prizes and internship opportunities"
output = generate_enhanced_prompt_final(user_input)

# Clean the output starting from the second index and print
cleaned_output_str = ' '.join(output["enhanced_prompt"][1:5])

# Print the cleaned output as a string
print("Cleaned Output:", cleaned_output_str)
