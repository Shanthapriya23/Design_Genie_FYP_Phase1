import spacy
import pandas as pd

# Load spaCy model for NLP processing
nlp = spacy.load('en_core_web_sm')

def preprocess_prompt(prompt):
    doc = nlp(prompt.lower())
    adjectives = [token.lemma_ for token in doc if token.pos_ == 'ADJ']
    objects = [token.lemma_ for token in doc if token.pos_ == 'NOUN']
    return adjectives, objects

def find_relevant_images(adjectives, objects, dataset):
    if dataset.empty:
        return []

    relevant_images = []
    for idx, row in dataset.iterrows():
        attributes = row['attributes']
        if all(adj in attributes for adj in adjectives) and all(obj in attributes for obj in objects):
            if pd.notna(row['ImageURL']) and pd.notna(row['ProductTitle']):
                relevant_images.append({
                    'title': row['ProductTitle'],
                    'image_url': row['ImageURL']
                })
    return relevant_images
