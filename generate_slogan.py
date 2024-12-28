import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
import os

# ============================
# 1. Initialize NLP Tools
# ============================

# Load spaCy model for NER
nlp = spacy.load('en_core_web_sm')

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ============================
# 2. Load and Prepare Dataset
# ============================

def load_dataset(csv_path):
    """Load the dataset from a CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The file {csv_path} does not exist.")
    df = pd.read_csv(csv_path)
    required_columns = {'company', 'Slogans', 'Category'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing columns in the dataset: {missing}")
    df.dropna(subset=['company', 'Slogans', 'Category'], inplace=True)
    return df.reset_index(drop=True)

# ============================
# 3. Preprocessing Functions
# ============================

def preprocess_text(text):
    """Preprocess the input text."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# ============================
# 4. Vectorization and Similarity
# ============================

def create_category_descriptions(df):
    """Create aggregated category descriptions by concatenating processed slogans."""
    return df.groupby('Category')['Slogans'].apply(lambda x: ' '.join(x)).to_dict()

def vectorize_categories(category_descriptions):
    """Vectorize category descriptions."""
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(list(category_descriptions.values())), vectorizer

# ============================
# 5. Model Initialization
# ============================

def load_model(model_name='sshleifer/distilbart-cnn-12-6'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# ============================
# 6. Categorization and Generation
# ============================

def categorize_prompt(prompt, vectorizer, category_vectors, categories):
    """Categorize the user prompt by computing similarity with category descriptions."""
    processed_prompt = preprocess_text(prompt)
    prompt_vector = vectorizer.transform([processed_prompt])
    similarities = cosine_similarity(prompt_vector, category_vectors).flatten()
    return categories[similarities.argmax()]

def generate_unique_slogan(prompt, category, tokenizer, model):
    """Generate a unique slogan based on the user prompt and identified category."""
    input_text = f"Generate a catchy slogan for: {prompt} Category: {category}"

    # Prepare to generate slogans
    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)

    # Generate slogan
    outputs = model.generate(
        inputs,
        max_length=50,
        min_length=10,
        num_beams=5,
        early_stopping=True,
        temperature=0.7,  # Control the randomness of predictions
        top_k=50,         # Limit the number of highest probability tokens
        top_p=0.95        # Nucleus sampling
    )
    
    generated_slogan = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()  # Clean output

    # Ensure uniqueness: Check if the generated slogan is identical or similar to the prompt
    if prompt.lower() in generated_slogan.lower() or cosine_similarity(vectorizer.transform([preprocess_text(prompt)]), vectorizer.transform([preprocess_text(generated_slogan)]))[0][0] > 0.5:
        # If too similar, regenerate
        return generate_unique_slogan(prompt, category, tokenizer, model)

    return generated_slogan

def main():
    csv_path = "slogans_combined_cleaned.csv"
    try:
        df = load_dataset(csv_path)
        print(f"Dataset loaded successfully with {len(df)} entries.")
    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        return

    # Preprocess dataset and create category descriptions
    df['Processed_Slogans'] = df['Slogans'].apply(preprocess_text)
    category_descriptions = create_category_descriptions(df)
    category_vectors, vectorizer = vectorize_categories(category_descriptions)
    categories = list(category_descriptions.keys())

    # Load model and tokenizer
    tokenizer, model = load_model()
    print("Model and tokenizer loaded successfully.")

    print("\n=== Slogan Generator ===")
    user_prompt = input("Enter a description of your event: ")

    # Categorize the prompt
    category = categorize_prompt(user_prompt, vectorizer, category_vectors, categories)
    print(f"Identified Category: {category}")

    # Generate a unique slogan
    slogan = generate_unique_slogan(user_prompt, category, tokenizer, model)
    print(f"Generated Slogan: {slogan}")

if __name__ == "__main__":
    main()
