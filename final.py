# image_processing.py
import os
import pandas as pd
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from torchvision import transforms
import spacy

# Load the Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Load the spaCy model with word vectors
nlp = spacy.load("en_core_web_md")  # Use the larger model with pre-trained vectors

# Function to match user prompt with labels in the dataset
def get_matching_images(prompt, df, image_folder,threshold=0.53):
    # Process the prompt using spaCy and filter out stopwords and punctuation
    prompt_doc = nlp(prompt.lower())
    prompt_tokens = [token.lemma_ for token in prompt_doc if not token.is_stop and not token.is_punct]

    matched_images = []

    for i, row in df.iterrows():
        # Process the label using spaCy and filter out stopwords and punctuation
        label_doc = nlp(row['label'].lower())
        label_tokens = [token.lemma_ for token in label_doc if not token.is_stop and not token.is_punct]
        # Create a document from label tokens
        label_text = ' '.join(label_tokens)
        label_doc = nlp(label_text)
        # Calculate cosine similarity between prompt and label
        if prompt_doc.has_vector and label_doc.has_vector:
            similarity = prompt_doc.similarity(label_doc)
            if similarity >= threshold:  # Only consider matches above the threshold
                matched_images.append(os.path.join(image_folder, row['File Name']))

    return matched_images

def load_images(image_paths, size=(200, 200)):
    images = []
    for img_path in image_paths:
        if os.path.exists(img_path):  # Check if the image exists
            print(f"Loading image: {img_path}")
            image = Image.open(img_path).convert('RGB')
            image = image.resize(size)  # Resize the image to the specified size
            images.append(image)
        else:
            print(f"Image not found: {img_path}")
    return images

def merge_images(images):
    if not images:
        return None
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)

    merged_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        merged_image.paste(img, (x_offset, 0))
        x_offset += img.width

    return merged_image

def generate_poster(merged_image, prompt):
    # Resize the image for efficiency
    max_size = (512, 512)  # Set maximum width and height
    merged_image.thumbnail(max_size)  # Resize while maintaining aspect ratio

    # Convert the merged image to a tensor for model processing
    preprocess = transforms.ToTensor()
    merged_tensor = preprocess(merged_image).unsqueeze(0)

    # Generate the poster using the text prompt and image conditioning
    generated_image = pipe(prompt, init_image=merged_tensor, guidance_scale=7.5).images[0]

    return generated_image

# Main execution flow
def stable_diffusion(user_prompt):
    # Path configurations
    csv_file = 'b_fin.csv'  # CSV file path
    image_folder = 'combined/combined'  # Folder where images are stored
    # Step 1: Load the CSV dataset
    df = pd.read_csv(csv_file)
    # Get matched image paths based on user input
    matched_image_paths = get_matching_images(user_prompt, df, image_folder)
    print("Matched image paths:", matched_image_paths)  # Check the matched image paths

    # Load the images
    images = load_images(matched_image_paths, size=(200, 200))  # Specify desired image size here

    # Merge all the images
    merged_image = merge_images(images)  # Merge all loaded images

    # Check if merging was successful
    if merged_image:
        # Display merged image
        merged_image.show()  # Show merged image for validation

        # Generate the poster
        poster = generate_poster(merged_image, user_prompt)
        #poster.show()  # Display the generated poster
        return poster
    else:
        print("No images to merge.")
        return None
    
