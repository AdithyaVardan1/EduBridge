from transformers import AutoTokenizer
import datasets

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

dataset = load_dataset("prsdm/MedQuad-phi2-1k")
tokenized_datasets = datasets.map(tokenize_function, batched=True)
from transformers import AutoModelForSequenceClassification

num_labels = 2  # Adjust based on your task
model = AutoModelForSequenceClassification.from_pretrained("microsoft/phi-2", num_labels=num_labels)
from transformers import TrainingArguments, Trainer
import intel_extension_for_pytorch as ipex

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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    # You can add a compute_metrics function here if you have specific metrics in mind
)

# Move model to IPEX device for optimized training
model.to(ipex.DEVICE)
