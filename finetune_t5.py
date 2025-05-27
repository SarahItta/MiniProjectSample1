from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load dataset
dataset = load_dataset('json', data_files='datasets/training_dataset.json')
print("Dataset loaded:", dataset)
dataset = dataset.filter(lambda x: x['task'] in ['summarization', 'question_generation'])
print("Filtered dataset:", dataset)
train_dataset = dataset['train']

# Split train/validation (80% train, 20% validation)
train_size = int(0.8 * len(train_dataset))
train_dataset, val_dataset = train_dataset.select(range(train_size)), train_dataset.select(range(train_size, len(train_dataset)))
print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

# Load tokenizer and model
try:
    tokenizer = T5Tokenizer.from_pretrained("./t5_finetuned")
    model = T5ForConditionalGeneration.from_pretrained("./t5_finetuned")
except Exception as e:
    print(f"Error loading ./t5_finetuned: {e}. Using t5-small instead.")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Preprocess
def preprocess_function(examples):
    print("Examples keys:", list(examples.keys()))
    inputs = []
    targets = []
    for i in range(len(examples['task'])):
        task = examples['task'][i]
        if task == 'summarization':
            inputs.append(examples['input_text'][i])
            targets.append(examples['summary'][i])
        elif task == 'question_generation':
            inputs.append(f"Generate a question for: {examples['context'][i]}")
            targets.append(examples['question'][i])
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./t5_finetuned_new',
    num_train_epochs=5,  # Increased for small dataset
    per_device_train_batch_size=2,  # CPU-friendly
    per_device_eval_batch_size=2,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=60,  # Multiple of eval_steps
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

# Train and save
trainer.train()
model.save_pretrained('./t5_finetuned_new')
tokenizer.save_pretrained('./t5_finetuned_new')