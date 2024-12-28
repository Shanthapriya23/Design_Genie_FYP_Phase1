from flask import Flask, request, render_template
import pandas as pd
from data_processing import preprocess_prompt, find_relevant_images
from prompt_enhancement import generate_enhanced_prompt
from trial5 import response
import requests
import base64
import wandb
import torch
import time

app = Flask(__name__)

# Load dataset
dataset = pd.read_csv("final_combined.csv")
dataset['attributes'] = dataset[['Gender', 'Category', 'SubCategory', 'ProductType', 'Colour', 'Usage']].apply(
    lambda x: ' '.join(x.dropna().astype(str)).lower(), axis=1)

# Initialize W&B for GPU utilization tracking
wandb.init(project="stable_diffusion_gpu_usage")

def call_stable_diffusion_api(prompt):
    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
    headers = {"Authorization": "Bearer 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"}

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

def call_magic_prompt_api(user_prompt):
    api_url = "https://api-inference.huggingface.co/models/Gustavosta/MagicPrompt-Stable-Diffusion"
    headers = {
        "Authorization": "Bearer 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"  # Replace with your Hugging Face API key
    }
    payload = {"inputs": user_prompt}

    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result[0]['generated_text']  # Adjust based on response structure
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def log_gpu_usage(action_name):
    """Log GPU memory usage and utilization to W&B for each action."""
    gpu_memory = torch.cuda.memory_allocated() / 1e6  # Convert to MB
    gpu_utilization = torch.cuda.utilization() if torch.cuda.is_available() else 0

    # Log GPU stats to W&B with action name
    wandb.log({
        "Action": action_name,
        "GPU Memory (MB)": gpu_memory,
        "GPU Utilization (%)": gpu_utilization
    })

@app.route('/', methods=['GET', 'POST'])
def index():
    relevant_images = []
    enhanced_prompt = None
    enhanced_prompt_nlp_magic = None
    further_enhanced_prompt = None
    magic_prompt = None
    magic_prompt_nlp = None
    generated_image = None
    generated_enhanced_image = None
    generated_magic_prompt_image = None
    generated_magic_prompt_image_nlp = None
    highlighted_button = None
    check_logical = None

    if request.method == 'POST':
        user_prompt = request.form['prompt']
        action = request.form['action']
        additional_details = request.form.get('additional_details')
        time = request.form.get('time')
        environment = request.form.get('environment')

        if action == 'enhanced_prompt':
            enhanced_prompt = generate_enhanced_prompt(user_prompt, time, environment)
            highlighted_button = 'enhanced_prompt'
            log_gpu_usage(action)

        elif action == 'check_logical':
            check_logical = response(user_prompt)
            highlighted_button = 'check_logical'
            log_gpu_usage(action)

        elif action == 'magic_prompt':
            magic_prompt = call_magic_prompt_api(user_prompt)
            highlighted_button = 'magic_prompt'
            if magic_prompt:
                print(f"Generated Magic Prompt: {magic_prompt}")
            log_gpu_usage(action)

        elif action == 'search_img':
            adjectives, objects = preprocess_prompt(user_prompt)
            relevant_images = find_relevant_images(adjectives, objects, dataset)
            log_gpu_usage(action)

        elif action == 'generate_normal_img':
            image_binary = call_stable_diffusion_api(user_prompt)
            if image_binary:
                generated_image = base64.b64encode(image_binary).decode('utf-8')
            log_gpu_usage(action)

        elif action == 'generate_enhanced_img':
            enhanced_prompt = generate_enhanced_prompt(user_prompt, time, environment)
            print(f"Generating image for enhanced prompt: {enhanced_prompt}")
            image_binary = call_stable_diffusion_api(enhanced_prompt)
            if image_binary:
                generated_enhanced_image = base64.b64encode(image_binary).decode('utf-8')
            highlighted_button = 'generate_enhanced_img'
            log_gpu_usage(action)

        elif action == 'generate_magic_prompt_img':
            magic_prompt = call_magic_prompt_api(user_prompt)
            if magic_prompt:
                print(f"Generated Magic Prompt: {magic_prompt}")
                image_binary = call_stable_diffusion_api(magic_prompt)
                if image_binary:
                    generated_magic_prompt_image = base64.b64encode(image_binary).decode('utf-8')
            highlighted_button = 'generate_magic_prompt_img'
            log_gpu_usage(action)

        elif action == 'generate_magic_prompt_img_nlp':
            magic_prompt_nlp = call_magic_prompt_api(user_prompt)
            enhanced_prompt_nlp_magic = generate_enhanced_prompt(magic_prompt_nlp, time, environment)
            if magic_prompt_nlp:
                print(f"API Enhanced Prompt: {magic_prompt_nlp}")
                print(f"NLP and API Enhanced Prompt: {enhanced_prompt_nlp_magic}")
                image_binary = call_stable_diffusion_api(enhanced_prompt_nlp_magic)
                if image_binary:
                    generated_magic_prompt_image_nlp = base64.b64encode(image_binary).decode('utf-8')
            highlighted_button = 'generate_magic_prompt_img_nlp'
            log_gpu_usage(action)

        elif action == 'additional_info':
            further_enhanced_prompt = generate_enhanced_prompt(user_prompt + " " + additional_details, time, environment)
            highlighted_button = 'further_enhance'
            print(f"Generating further enhanced prompt: {further_enhanced_prompt}")
            image_binary = call_stable_diffusion_api(further_enhanced_prompt)
            if image_binary:
                generated_magic_prompt_image_nlp = base64.b64encode(image_binary).decode('utf-8')
            log_gpu_usage(action)

    return render_template('index.html',
                           images=relevant_images,
                           check_logical=check_logical,
                           enhanced_prompt=enhanced_prompt,
                           enhanced_prompt_nlp_magic=enhanced_prompt_nlp_magic,
                           further_enhanced_prompt=further_enhanced_prompt,
                           magic_prompt=magic_prompt,
                           magic_prompt_nlp=magic_prompt_nlp,
                           generated_image=generated_image,
                           generated_enhanced_image=generated_enhanced_image,
                           generated_magic_prompt_image=generated_magic_prompt_image,
                           generated_magic_prompt_image_nlp=generated_magic_prompt_image_nlp,
                           highlighted_button=highlighted_button)

if __name__ == '__main__':
    app.run(debug=True)
