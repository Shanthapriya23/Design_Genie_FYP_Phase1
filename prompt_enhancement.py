import json
import random
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

nltk.download('vader_lexicon')

nlp = spacy.load('en_core_web_sm')

sid = SentimentIntensityAnalyzer()

# Initializing transformers pipelines
classifier = pipeline("zero-shot-classification")
text_generator = pipeline("text-generation", model="gpt2")

# Improved entity classification
def classify_entity(entity):
    # Using zero-shot classification for more flexible entity classification
    categories = ["human", "animal", "plant", "object", "concept", "place"]
    result = classifier(entity, categories)
    return result['labels'][0]  # Return the most likely category

# Enhanced POS tagging and NER
def get_entities_and_attributes(prompt):
    doc = nlp(prompt)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    verbs = [token.text for token in doc if token.pos_ == "VERB"]
    return entities, nouns, adjectives, verbs

# Improved sentiment analysis (Combine VADER with a more nuanced model)
def deep_sentiment_analysis(prompt):
    vader_sentiment = sid.polarity_scores(prompt)

    # Use a pre-trained model for more nuanced sentiment analysis
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    inputs = sentiment_tokenizer(prompt, return_tensors="pt")
    outputs = sentiment_model(**inputs)

    # Combine VADER and transformer model results
    sentiment_score = (vader_sentiment['compound'] + outputs.logits.softmax(dim=1)[0][1].item()) / 2

    if sentiment_score > 0.05:
        sentiment_category = 'positive'
    elif sentiment_score < -0.05:
        sentiment_category = 'negative'
    else:
        sentiment_category = 'neutral'

    return sentiment_category, sentiment_score

# Load attributes from JSON file
def load_attributes():
    with open('attributes.json', 'r') as file:
        return json.load(file)

# Improved attribute association based on entity type, sentiment, and similarity
def associate_attributes(entity, entity_type, sentiment_category, threshold=0.6):
    attributes = load_attributes()
    
    # Get possible attribute words for the entity type and sentiment category
    attribute_words = attributes.get(entity_type, attributes["object"]).get(sentiment_category, [])

    # Find the most semantically similar attributes using Spacy
    entity_doc = nlp(entity.lower())
    similar_attributes = []

    for word in attribute_words:
        word_doc = nlp(word.lower())
        if entity_doc.has_vector and word_doc.has_vector:
            similarity = entity_doc.similarity(word_doc)
            if similarity >= threshold:
                similar_attributes.append(word)
    # If similar attributes are found
    if similar_attributes:
        return random.sample(similar_attributes, min(3, len(similar_attributes)))
    else:
        # If no similar attributes are found, return a random sample of up to 3 from attribute_words
        return random.sample(attribute_words, min(3, len(attribute_words)))


# Load atmosphere elements from JSON file
def load_atmosphere_elements():
    with open('atmosphere_elements.json', 'r') as file:
        return json.load(file)

# Load weather data from JSON file
def load_weather():
    with open('weather.json', 'r') as file:
        return json.load(file)
    
# 1. Dynamic time description based on input
def describe_time(time):
    if not time:  # Check if time is empty
        return "unknown time"
    hour = int(time.split(":")[0])  # Extract hour from "HH:MM" format
    if 5 <= hour < 8:
        return "early morning"
    elif 8 <= hour < 12:
        return "morning"
    elif 12 <= hour < 15:
        return "sunny afternoon"
    elif 15 <= hour < 18:
        return "late afternoon"
    elif 18 <= hour < 21:
        return "evening"
    elif hour < 5:
        return "midnight"
    else:
        return "night"

# Extract similar words using Spacy
def extract_similar_words(input_text, attributes, threshold=0.6):
    doc = nlp(input_text.lower())
    similar_words = set()

    for category, attribute_types in attributes.items():
        for sentiment, words in attribute_types.items():
            for word in words:
                word_doc = nlp(word.lower())
                for token in doc:
                    if token.has_vector and word_doc.has_vector:
                        similarity = token.similarity(word_doc)
                        if similarity >= threshold and token.text.lower() != word.lower():
                            similar_words.add(word)

    return list(similar_words)

# 2. Dynamically generate environment details
def generate_environment_description(environment):
    attributes = load_attributes()
    similar_words = extract_similar_words(environment, attributes)
    
    # Incorporate similar words dynamically based on environment input
    if similar_words:
        return f"a {', '.join(similar_words)} {environment}"
    else:
        return f"a {environment}"
       
# Generate dynamic scene and atmosphere
def generate_dynamic_scene(sentiment_category,time,environment):
    if time:  # Check if time is not empty
        time_description = describe_time(time) 
    else:
        time_description = "midday"
    weather_data = load_weather()
    weather = random.choice(weather_data.get(sentiment_category, ["changing weather"]))
    # Environment description from user input
    environment_description = generate_environment_description(environment)
    # Create a dynamic scene
    atmosphere = f"{time_description} with {weather} in {environment_description}."
    return atmosphere

# Map sentiment to color palette
def load_palettes():
    with open('palettes.json', 'r') as file:
        return json.load(file)

def map_sentiment_to_color_palette(sentiment_category, sentiment_score):
    palettes = load_palettes()
    #print("Loaded palettes:", palettes)
    #print("Sentiment category:", sentiment_category)
    # Check the available keys in palettes
    print("Available keys:", palettes["palettes"].keys())

    # Retrieve the color palette for the sentiment category
    palette = palettes["palettes"].get(sentiment_category, palettes["palettes"].get("neutral", []))

    if not palette:
        palette = ["grey"]

    return random.sample(palette, 3)

# Generate enhanced prompt
def generate_enhanced_prompt(user_prompt,time=None,environment=None):
    print("time: ",time)
    print("environment:",environment)
    # Step 1: Deep sentiment analysis
    sentiment_category, sentiment_score = deep_sentiment_analysis(user_prompt)

    # Step 2: Entity extraction and classification
    entities, nouns, adjectives, verbs = get_entities_and_attributes(user_prompt)

    # Step 3: Enhanced entity attributes
    enhanced_entities = []
    for entity, entity_type in entities:
        attributes = associate_attributes(entity, classify_entity(entity), sentiment_category)
        if attributes:
            attributes_str = ', '.join(attributes)
            enhanced_entities.append(f"Include elements that evoke feelings of {attributes_str} in {entity}.")

    # If no entities were found, use the nouns from the original prompt
    if not enhanced_entities:
        for noun in nouns:
            attributes = associate_attributes(noun, classify_entity(noun), sentiment_category)
            if attributes:
                attributes_str = ', '.join(attributes)
                enhanced_entities.append(f"Include elements that evoke feelings of {attributes_str} in {noun}.")

    # Combine the enhanced entities into a single prompt
    enhanced_prompt = ' '.join(enhanced_entities)

    # Step 4: Dynamic scene generation
    scene = generate_dynamic_scene(sentiment_category, time or "12:00", environment or "a generic location")
    print("scene: ", scene)
    # Step 5: Color palette selection
    color_palette = map_sentiment_to_color_palette(sentiment_category, sentiment_score)

    # Step 6: Combine elements into an enhanced prompt
    enhanced_prompt = user_prompt+". "
    enhanced_prompt += f"Create an image of {' and '.join(enhanced_entities)} {scene} "
    enhanced_prompt += f"Use a color palette of {', '.join(color_palette)}. "
    enhanced_prompt += f"{scene}."
        
    # Step 7: Add action and mood
    if verbs:
        action = random.choice(verbs)
        enhanced_prompt += f"The scene should convey a sense of {action}. "
    enhanced_prompt += f"The overall mood should be {sentiment_category}. "

    # Step 8: Add artistic style suggestion
    art_styles = ["photorealistic", "impressionistic", "surrealist", "abstract", "digital art", "oil painting"]
    enhanced_prompt += f"Render the image in a {random.choice(art_styles)} style."

    return enhanced_prompt
