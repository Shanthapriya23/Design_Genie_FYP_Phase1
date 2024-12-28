# clean_slogan_dataset.py

import pandas as pd
import re

def clean_slogan_text(text):
    # Remove non-ASCII characters
    text = text.encode('ascii', errors='ignore').decode('utf-8')

    # Remove unwanted punctuation (except for basic punctuation)
    text = re.sub(r'[^\w\s\!\?\.]', '', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text

def clean_dataset(input_csv, output_csv):
    """
    Cleans the slogan dataset by handling encoding issues, removing bad lines,
    dropping rows with missing values, and cleaning the slogan text.

    Args:
        input_csv (str): Path to the original CSV file.
        output_csv (str): Path to save the cleaned CSV file.
    """
    try:
        print(f"Loading dataset from {input_csv}...")
        # Read the CSV file with UTF-8 encoding, skipping bad lines
        df = pd.read_csv(input_csv, encoding='utf-8', on_bad_lines='skip')
        print(f"Dataset loaded successfully with {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: The file {input_csv} does not exist.")
        return
    except pd.errors.ParserError as e:
        print(f"ParserError while reading {input_csv}: {e}")
        return

    # Display initial data info
    print("\nInitial DataFrame Info:")
    print(df.info())
    print("\nFirst 5 Rows:")
    print(df.head())

    # Drop rows with missing values in 'Slogans' and 'Category'
    initial_row_count = len(df)
    df.dropna(subset=['Slogans', 'Category'], inplace=True)
    cleaned_row_count = len(df)
    print(f"\nDropped {initial_row_count - cleaned_row_count} rows due to missing 'Slogans' or 'Category'.")

    # Clean the 'Slogans' column
    print("\nCleaning 'Slogans' column...")
    df['Slogans'] = df['Slogans'].apply(clean_slogan_text)

    # Optionally, clean other text columns if necessary (e.g., 'Company')
    if 'company' in df.columns:
        print("Cleaning 'company' column...")
        df['company'] = df['company'].apply(clean_slogan_text)

    # Optionally, standardize 'Category' names (e.g., capitalize first letter)
    if 'Category' in df.columns:
        print("Standardizing 'Category' names...")
        df['Category'] = df['Category'].str.strip().str.lower().str.capitalize()

    # Display cleaned data info
    print("\nCleaned DataFrame Info:")
    print(df.info())
    print("\nFirst 5 Cleaned Rows:")
    print(df.head())

    # Save the cleaned DataFrame to a new CSV file
    print(f"\nSaving cleaned dataset to {output_csv}...")
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print("Dataset cleaned and saved successfully.")

if __name__ == "__main__":
    # Define input and output CSV file paths
    input_csv = 'slogans_combined.csv'          # Replace with your original CSV file path
    output_csv = 'slogans_combined_cleaned.csv' # Desired path for the cleaned CSV

    clean_dataset(input_csv, output_csv)
