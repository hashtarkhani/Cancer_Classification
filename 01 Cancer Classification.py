import pandas as pd
import openai
import os
import difflib

# Load diagnosis data
data_path = 'path_to_data/LLM_Multimodel_Predicted.csv'
data = pd.read_csv(data_path)

# Predefined cancer diagnosis categories
categories = [
    "Benign", "Breast", "Lung or thoracic", "Prostate", "Gynecologic",
    "Head and Neck", "Gastrointestinal", "Central nervous system", "Metastasis",
    "Skin", "Soft tissue", "Hematologic", "Genitourinary", "Unknown"
]

# Set OpenAI API key
openai.api_key = 'your_api_key_here'  # Replace with your actual key

# Function to categorize diagnosis using ChatGPT
def categorize_with_chatgpt(text, model):
    prompt = (
        f"Given the following ICD-10 description or treatment note for a radiation therapy patient: "
        f"'{text}', determine the most appropriate category from this predefined list: {', '.join(categories)}. "
        "Respond with only the exact category name from the list, without any additional explanation or punctuation."
    )
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that categorizes cancer diagnosis data."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

# Function to clean model predictions
def clean_prediction(value, valid_categories):
    if isinstance(value, str):
        value = value.strip().lower()
        for cat in valid_categories:
            if value in cat.lower():
                return cat
        closest = difflib.get_close_matches(value, valid_categories, n=1)
        return closest[0] if closest else "No match"
    return "No match"

# Categorize using ChatGPT (GPT-3.5 and GPT-4)
data['GPT3.5'] = data['Code'].apply(lambda x: categorize_with_chatgpt(x, "gpt-3.5-turbo"))
data['GPT4'] = data['Code'].apply(lambda x: categorize_with_chatgpt(x, "gpt-4o"))

# Clean and standardize outputs
data['GPT3.5_Clean'] = data['GPT3.5'].apply(lambda x: clean_prediction(x, categories))
data['GPT4_Clean'] = data['GPT4'].apply(lambda x: clean_prediction(x, categories))

# Compute classification accuracy
def compute_accuracy(df, predicted_col, reference_col='Diagnosis'):
    return (df[predicted_col] == df[reference_col]).mean() * 100

# Separate data by type: free-text vs ICD-coded
free_text_data = data[data['Freq'] == 1]          # Unique, free-text diagnosis entries
icd_coded_data = data[data['Freq'] != 1]          # Structured, ICD-coded diagnosis entries

# Accuracy for ICD-coded entries
accuracy_icd = {
    "GPT-3.5 (ICD-coded)": compute_accuracy(icd_coded_data, 'GPT3.5_Clean'),
    "GPT-4 (ICD-coded)": compute_accuracy(icd_coded_data, 'GPT4_Clean')
}

# Accuracy for free-text entries
accuracy_free_text = {
    "GPT-3.5 (Free-text)": compute_accuracy(free_text_data, 'GPT3.5_Clean'),
    "GPT-4 (Free-text)": compute_accuracy(free_text_data, 'GPT4_Clean')
}

# Save final results

data.to_csv('path_to_output/FinalResults.csv', index=False)
