from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import intel_extension_for_pytorch as ipex

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

# Ensure the tokenizer has a pad token, use the eos_token if pad_token is not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # Use the tokenizer directly with padding and truncation
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Load dataset
dataset = load_dataset("prsdm/MedQuad-phi2-1k")

# Apply tokenization function to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load model for sequence classification with the specified number of labels
model = AutoModelForSequenceClassification.from_pretrained("microsoft/phi-2", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    # Define compute_metrics function for evaluation metrics if necessary
)

# No need to explicitly move the model to IPEX device, just ensure it's on CPU
model.to("cpu")

# Start training
trainer.train()
