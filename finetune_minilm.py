from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from datasets import load_dataset

# Load dataset
dataset = load_dataset('json', data_files='datasets/training_dataset.json')
dataset = dataset.filter(lambda x: x['task'] == 'semantic_similarity')
train_data = dataset['train']
print(f"Dataset size: {len(train_data)}")

# Split train/validation
train_size = int(0.8 * len(train_data))
train_data, val_data = train_data.select(range(train_size)), train_data.select(range(train_size, len(train_data)))
print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}")

# Load model
try:
    model = SentenceTransformer("./minilm_finetuned")
except Exception as e:
    print(f"Error loading ./minilm_finetuned: {e}. Using all-MiniLM-L6-v2 instead.")
    model = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare examples
train_examples = [InputExample(texts=[ex['sentence1'], ex['sentence2']], label=ex['score']) for ex in train_data]
val_examples = [InputExample(texts=[ex['sentence1'], ex['sentence2']], label=ex['score']) for ex in val_data]

# DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

# Loss
train_loss = losses.CosineSimilarityLoss(model=model)

# Evaluator
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name='sts-val')

# Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=10,
    evaluator=evaluator,
    evaluation_steps=20,
    output_path='./minilm_finetuned_new'
)

# Save
model.save('./minilm_finetuned_new')