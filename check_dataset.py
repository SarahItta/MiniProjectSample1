import json

with open('datasets/training_dataset.json', 'r') as f:
    data = json.load(f)

print(f"Total examples: {len(data)}")
print(f"Summarization: {sum(1 for x in data if x['task'] == 'summarization')}")
print(f"Question Generation: {sum(1 for x in data if x['task'] == 'question_generation')}")
print(f"Semantic Similarity: {sum(1 for x in data if x['task'] == 'semantic_similarity')}")

# Print a few examples
for i, item in enumerate(data[:5]):
    print(f"Example {i}: {item}")