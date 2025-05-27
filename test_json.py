import json

with open("flashcard_dataset.json", "r") as f:
    data = json.load(f)
print(data)