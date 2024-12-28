from flask import Flask, request, render_template,jsonify,send_file
import pandas as pd
from data_processing import preprocess_prompt, find_relevant_images
from prompt_enhancement import generate_enhanced_prompt
from slogan_generation import generate_slogan_final
from slogan_add import generate_image_with_slogan
from final import stable_diffusion
import requests
import base64
import random
from PIL import Image 
import google.generativeai as genai
import os
import logging
import io
from gpt import response
import time
import requests


app = Flask(__name__)

# Load dataset
dataset = pd.read_csv("final_combined.csv")
dataset['attributes'] = dataset[['Gender', 'Category', 'SubCategory', 'ProductType', 'Colour', 'Usage']].apply(
    lambda x: ' '.join(x.dropna().astype(str)).lower(), axis=1)


def call_stable_diffusion_api(prompt):
    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
    headers = {"Authorization": "Bearer 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  "}

    payload = {
        "inputs": prompt,
        "parameters": {
            "width": 512,  
            "height": 512  
        },
        "options": {"wait_for_model": True}
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.content
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def call_magic_prompt_api(user_prompt, retries=5, delay=5):
    api_url = "https://api-inference.huggingface.co/models/Gustavosta/MagicPrompt-Stable-Diffusion"
    headers = {
        "Authorization": "Bearer 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"  # Replace with your Hugging Face API key
    }
    payload = {"inputs": user_prompt}

    for attempt in range(retries):
        response = requests.post(api_url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            return result[0]['generated_text']  # Adjust based on response structure
        elif response.status_code == 503:
            error_info = response.json()
            estimated_time = error_info.get("estimated_time", delay)
            print(f"Attempt {attempt+1}: Model busy, retrying in {estimated_time} seconds.")
            time.sleep(estimated_time)
            delay *= 2  # Progressive backoff
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    print("Model could not load after multiple attempts.")
    return None


@app.route('/', methods=['GET', 'POST'])
def index():
    user_prompt = ""
    relevant_images = []
    slogans_data = []
    enhanced_prompt = None
    enhanced_prompt_nlp_magic = None
    further_enhanced_prompt = None
    magic_prompt = None
    magic_prompt_nlp = None
    generated_image = None 
    generated_enhanced_image = None  
    generated_magic_prompt_image = None
    generated_magic_prompt_image_nlp = None
    generated_poster_image = None
    final_poster_image = None
    highlighted_button = None
    check_logical = None    
    error=None
    
    if request.method == 'POST':
        user_prompt = request.form['prompt']
        action = request.form['action']
        additional_details = request.form.get('additional_details')
        time = request.form.get('time')
        environment = request.form.get('environment')

        if action == 'enhanced_prompt':
            enhanced_prompt = generate_enhanced_prompt(user_prompt,time,environment)
            highlighted_button = 'enhanced_prompt'

        elif action == 'check_logical':
            check_logical = response(user_prompt)
            highlighted_button = 'check_logical'

        elif action == 'magic_prompt':
            magic_prompt = call_magic_prompt_api(user_prompt)
            highlighted_button = 'magic_prompt'
            if magic_prompt:
                print(f"Generated Magic Prompt: {magic_prompt}")
        
        elif action == 'generate_normal_img':
            image_binary = call_stable_diffusion_api(user_prompt)
            if image_binary:
                generated_image = base64.b64encode(image_binary).decode('utf-8')

        elif action == 'generate_enhanced_img':
            enhanced_prompt = generate_enhanced_prompt(user_prompt,time,environment)
            print(f"Generating image for enhanced prompt: {enhanced_prompt}")
            image_binary = call_stable_diffusion_api(enhanced_prompt)
            if image_binary:
                generated_enhanced_image = base64.b64encode(image_binary).decode('utf-8')
            highlighted_button = 'generate_enhanced_img'

        elif action == 'generate_magic_prompt_img':
            magic_prompt = call_magic_prompt_api(user_prompt)
            if magic_prompt:
                print(f"Generated Magic Prompt: {magic_prompt}")
                image_binary = call_stable_diffusion_api(magic_prompt)
                if image_binary:
                    generated_magic_prompt_image = base64.b64encode(image_binary).decode('utf-8')
            highlighted_button = 'generate_magic_prompt_img'

        elif action == 'generate_magic_prompt_img_nlp':
            magic_prompt_nlp = call_magic_prompt_api(user_prompt)
            enhanced_prompt_nlp_magic = generate_enhanced_prompt(magic_prompt_nlp,time,environment)
            if magic_prompt_nlp:             
                print(f"API Enhanced Prompt: {magic_prompt_nlp}")
                print(f"NLP and API Enhanced Prompt: {enhanced_prompt_nlp_magic}")
                image_binary = call_stable_diffusion_api(enhanced_prompt_nlp_magic)
                if image_binary:
                    generated_magic_prompt_image_nlp = base64.b64encode(image_binary).decode('utf-8')
            highlighted_button = 'generate_magic_prompt_img_nlp'
            
        elif action=='additional_info':
            further_enhanced_prompt = generate_enhanced_prompt(user_prompt+" "+additional_details,time,environment)
            highlighted_button = 'further_enhance'
            print(f"Generating further enhanced prompt: {further_enhanced_prompt}")
            image_binary = call_stable_diffusion_api(further_enhanced_prompt)
            if image_binary:
                generated_magic_prompt_image_nlp = base64.b64encode(image_binary).decode('utf-8')
        
        elif action == 'feed_img_sd':
            try:
                # Call the stable_diffusion function from image_processing.py
                poster = stable_diffusion(user_prompt)
                if poster:
                    # Convert PIL Image to bytes
                    buffered = io.BytesIO()
                    poster.save(buffered, format="PNG")
                    generated_poster_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    highlighted_button = 'feed_img_sd'
                    print("Poster generated successfully.")
                else:
                    error = "Failed to generate poster from the dataset images."
                    print("Poster generation returned None.")
            except Exception as e:
                error = "An unexpected error occurred while generating the poster."
                print(f"Error in 'feed_img_sd' action: {e}")
        elif action == 'generate_slogan':
            # Generate slogan using the slogan generation module
            slogans_data = generate_slogan_final(user_prompt)
            highlighted_button = 'generate_slogan'
    return render_template('index.html', 
                           images=relevant_images, 
                           enhanced_prompt=enhanced_prompt,
                           enhanced_prompt_nlp_magic = enhanced_prompt_nlp_magic,
                           further_enhanced_prompt = further_enhanced_prompt,
                           magic_prompt = magic_prompt,
                           magic_prompt_nlp = magic_prompt_nlp,
                           generated_image=generated_image,
                           generated_enhanced_image=generated_enhanced_image,
                           generated_magic_prompt_image=generated_magic_prompt_image,
                           generated_magic_prompt_image_nlp=generated_magic_prompt_image_nlp,
                           highlighted_button=highlighted_button,
                           generated_poster_image=generated_poster_image,
                           check_logical = check_logical,
                           final_poster_image = final_poster_image,
                           slogans_data=slogans_data,
                           prompt = user_prompt)

@app.route('/generate_slogan_poster', methods=['POST'])
def generate_slogan_poster():
    final_poster_image = None
    time = request.form.get('time')
    environment = request.form.get('environment')
    slogan = request.form.get('slogan')
    prompt = request.form.get('prompt')

    #enhanced_prompt = call_magic_prompt_api(prompt)
    #print(f"Generating enhanced prompt: {enhanced_prompt}")
    generated_image = call_stable_diffusion_api(prompt) 
    generated_image = Image.open(io.BytesIO(generated_image))  # Convert the generated image to PIL format if needed

    font_path = "Roboto-BlackItalic.ttf"  # Specify the actual path to the font you wish to use

    # Adding slogan to the poster
    image_with_slogan = generate_image_with_slogan(generated_image, slogan, font_path)

    # Return the image with the slogan
    if image_with_slogan:
        return send_file(image_with_slogan, mimetype='image/png')
    else:
        return jsonify({"error": "No space found for slogan"})
    
if __name__ == '__main__':
    app.run(debug=True)
