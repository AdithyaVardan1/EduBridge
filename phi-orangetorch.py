import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cpu")  # Assuming no GPU

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("rhysjones/phi-2-orange").to(device)
tokenizer = AutoTokenizer.from_pretrained("rhysjones/phi-2-orange")

# User prompt
user_prompt = input("Enter your question: ")

# Construct simplified ChatML prompt
prompt = f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n<|im_start|>assistant"

# Generate response with truncated max length and lower temperature
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
inputs = {k: v.to(device) for k, v in inputs.items()}
outputs = model.generate(
    **inputs,
    max_length=200,  # Truncated for shorter output
    do_sample=True,
    top_p=0.92,
    temperature=0.7  # Lower temperature for potentially faster generation
)

# Decode and truncate based on full stops
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
stop_indices = [i for i, char in enumerate(text) if char == "."]
truncated_text = text[:stop_indices[2] + 1]  # Truncate at the third full stop

# Remove redundant system prompts
truncated_text = truncated_text.split("<|im_end|>")[1].strip()

print(truncated_text)
