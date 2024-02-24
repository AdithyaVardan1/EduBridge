
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set device to CPU (assuming no GPU availability)
device = torch.device("cpu")

# Load Gemma model and tokenizer
model_name = "google/gemma-2b-it"  # Specify the desired Gemma model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

user_prompt = input("Enter your prompt: ")

# Tokenize the user input
inputs = tokenizer(user_prompt, return_tensors="pt", return_attention_mask=False)

# Move input tensors to the CPU
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate text
start_time = time.time()
outputs = model.generate(
    **inputs,
    max_length=200,
    do_sample=True,
    top_p=0.92,
    temperature=0.85
)
end_time = time.time()

# Decode and print the generated text
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: {text}")
print(f"Inference time: {end_time - start_time:.2f} seconds")
