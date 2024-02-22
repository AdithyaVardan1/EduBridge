import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from tqdm import tqdm
import torch


MODEL_NAME = 'LLaMA-7B'  

# Load and preprocess the dataset
df = pd.read_csv('questions_and_answers.csv')
df.dropna(subset=['Question', 'Answer'], inplace=True)  

# Define a custom dataset class
class QADataset(Dataset):
    def __init__(self, tokenizer, questions, answers):
        self.tokenizer = tokenizer
        self.questions = questions
        self.answers = answers

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        inputs = self.tokenizer.encode_plus(
            question,
            answer,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

# Initialize tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = QADataset(tokenizer, list(df['Question']), list(df['Answer']))

# Create data loader
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Move model to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Fine-tuning loop
model.train()
for epoch in range(1):  # Replace with the number of epochs you want
    loop = tqdm(loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

# Save the fine-tuned model
model.save_pretrained('D:\\Bolt Vit Vellore\\finetune.py')
