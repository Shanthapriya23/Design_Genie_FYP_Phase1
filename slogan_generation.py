import requests
import random
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

API_KEY = "AIzaSyD96PJQN2yCDlHkdnDn4xG1Hb85JMRaJrQ"

genai.configure(api_key=API_KEY)


semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_slogan_dataset(csv_path='slogans_combined.csv'):
    return pd.read_csv(csv_path)

def preprocess_slogan_dataset(dataset):
    categories = dataset['Category'].dropna().astype(str).unique().tolist() 
    category_slogans = {}
    for category in categories:
        category_slogans[category] = dataset[dataset['Category'].astype(str) == category]['Slogans'].dropna().tolist()
    return category_slogans

def get_slogans_by_category(category_slogans, category):
    return category_slogans.get(category, [])

# Function to calculate semantic similarity between the prompt and slogans
def calculate_semantic_similarity(prompt, slogans):
    prompt_embedding = semantic_model.encode([prompt])[0]
    slogan_embeddings = semantic_model.encode(slogans)

    similarities = cosine_similarity([prompt_embedding], slogan_embeddings)[0]
    
    # Create a list of (slogan, similarity) tuples
    ranked_slogans = list(zip(slogans, similarities))
    
    ranked_slogans.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_slogans

def generate_catchy_slogans(user_prompt, category_slogans):
    generation_config = {
        "temperature": 0.8, 
        "max_output_tokens": 250,  
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    )

    category = determine_category(user_prompt, category_slogans.keys())
    if not category:
        category = 'General'

    example_slogans = random.sample(
        get_slogans_by_category(category_slogans, category),
        min(3, len(get_slogans_by_category(category_slogans, category)))
    )

    prompt = f"""Generate 10 unique and catchy slogans each of 3 words maximum (short and precise) for an event with the following description: '{user_prompt}'.
Consider these example slogans in the '{category}' category:
{chr(10).join([f"- {slogan}" for slogan in example_slogans])}

Provide exactly 10 slogans, one per line, numbered from 1-10. Make each slogan distinct and memorable."""

    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)

    # Process the response
    return clean_output(response.text.strip())

def clean_output(text):
    slogans = []
    for line in text.split('\n'):
        cleaned_line = line.strip()
        cleaned_line = ''.join([char for char in cleaned_line if not char.isdigit()])
        cleaned_line = cleaned_line.replace('*', '').replace('-', '').replace(')', '').replace('.', '').strip()
        
        if len(cleaned_line) > 5:
            slogans.append(cleaned_line)
    slogans = slogans[1:11]
    return slogans

def determine_category(user_prompt, categories):
    user_prompt_lower = str(user_prompt).lower()
    for category in categories:
        if isinstance(category, str) and category.lower() in user_prompt_lower:
            return category
    return None

# Main function to generate and rank slogans
def generate_slogan_final(user_input):
    dataset = load_slogan_dataset()
    category_slogans = preprocess_slogan_dataset(dataset)
    generated_slogans = generate_catchy_slogans(user_input, category_slogans)

    ranked_slogans = calculate_semantic_similarity(user_input, generated_slogans)
    
    slogans_data = [{"slogan": slogan, "score": score} for slogan, score in ranked_slogans]
    print(slogans_data)
    return slogans_data
