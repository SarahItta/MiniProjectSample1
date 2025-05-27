import json

def validate_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    for i, item in enumerate(data):
        if item['task'] == 'summarization':
            if len(item['input_text'].split()) < 20:
                print(f"Warning: Short input at index {i}")
            if not item['summary']:
                print(f"Error: Missing summary at index {i}")
        elif item['task'] == 'question_generation':
            if not item['context'] or not item['question'] or not item['answer']:
                print(f"Error: Missing fields at index {i}")
        elif item['task'] == 'semantic_similarity':
            if not (0.0 <= item['score'] <= 1.0):
                print(f"Error: Invalid score at index {i}")

validate_dataset('datasets/training_dataset.json')