import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load the dataset
def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The file '{csv_path}' does not exist.")
    
    df = pd.read_csv(csv_path)
    
    # Verify required columns exist
    required_columns = {'company', 'Slogans', 'Category'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing columns in dataset: {missing}")
    
    return df

# Clean the dataset by removing non-string 'Slogans'
def clean_dataset(df):
    # Remove rows where 'Slogans' is not a string
    df_cleaned = df[df['Slogans'].apply(lambda x: isinstance(x, str))].copy()
    
    return df_cleaned

# Preprocess the slogans
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)
    keywords = [word for word, tag in tagged if tag.startswith('NN') or tag.startswith('VB')]
    return ' '.join(keywords)

# Generate a slogan using GPT-2
def generate_slogan(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=30, num_return_sequences=1)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated

# Main function to process the dataset and generate slogans
def main():
    # Load and clean the dataset
    csv_path = 'slogans_combined.csv'  # Ensure this file is in the same directory
    df = load_dataset(csv_path)
    df_cleaned = clean_dataset(df)
    print(f"Original dataset size: {len(df)}")
    print(f"Cleaned dataset size: {len(df_cleaned)}")

    # Apply preprocessing
    df_cleaned['Processed_Slogans'] = df_cleaned['Slogans'].apply(preprocess_text)

    # Load the pre-trained GPT-2 model and tokenizer
    model_name = 'gpt2'  # You can change this to 'gpt2-medium', 'gpt2-large', etc.
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Generate a slogan for each category in the cleaned dataset
    for index, row in df_cleaned.iterrows():
        prompt = f"Generate a catchy slogan for {row['Category']}:"
        generated_slogan = generate_slogan(prompt, model, tokenizer)
        print(f"Generated Slogan for {row['Category']}: {generated_slogan}")

# Run the main function
if __name__ == "__main__":
    main()
