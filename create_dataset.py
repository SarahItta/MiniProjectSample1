import pandas as pd
import json
import re

# Preprocess text function (from your app.py)
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text

# Dataset entry for photosynthesis
input_text = """
Photosynthesis is a vital process where green plants, algae, and some bacteria convert sunlight, carbon dioxide, and water into glucose and oxygen. This process occurs in chloroplasts, which contain chlorophyll, a pigment that absorbs light energy. Photosynthesis consists of two main stages: the light-dependent reactions, which produce ATP and NADPH, and the Calvin cycle, which uses these products to fix carbon dioxide into glucose.
"""

summary = """
Photosynthesis is the process by which green plants, algae, and some bacteria use sunlight, carbon dioxide, and water to produce glucose and oxygen. It occurs in chloroplasts, where chlorophyll absorbs light energy. The process involves light-dependent reactions, generating ATP and NADPH, and the Calvin cycle, which fixes carbon dioxide into glucose.
"""

keywords = ["photosynthesis", "chlorophyll", "chloroplasts", "light-dependent reactions", "Calvin cycle"]

flashcards = [
    {"front": "What is photosynthesis?", "back": "Photosynthesis is a vital process where green plants, algae, and some bacteria convert sunlight, carbon dioxide, and water into glucose and oxygen."},
    {"front": "What is the role of chlorophyll?", "back": "Chlorophyll is a pigment in chloroplasts that absorbs light energy for photosynthesis."},
    {"front": "What are the two main stages of photosynthesis?", "back": "The light-dependent reactions and the Calvin cycle."}
]

questions = [
    {"question": "Where does photosynthesis occur?", "answer": "In chloroplasts."},
    {"question": "What does the Calvin cycle do?", "answer": "It fixes carbon dioxide into glucose."},
    {"question": "What are the products of the light-dependent reactions?", "answer": "ATP and NADPH."}
]

# Save summarization dataset (for T5)
summarization_data = [{"input_text": preprocess_text(input_text), "summary": summary}]
summarization_df = pd.DataFrame(summarization_data)
summarization_df.to_csv("summarization_dataset.csv", index=False)
print("Summarization dataset saved to summarization_dataset.csv")

# Save flashcard/keyword dataset (for SentenceTransformer)
flashcard_data = [{
    "text": preprocess_text(input_text),
    "keywords": keywords,
    "flashcards": flashcards,
    "questions": questions
}]
with open("flashcard_dataset.json", "w") as f:
    json.dump(flashcard_data, f, indent=2)
print("Flashcard dataset saved to flashcard_dataset.json")
