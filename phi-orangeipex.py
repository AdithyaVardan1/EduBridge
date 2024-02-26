import torch
import intel_extension_for_pytorch as ipex
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("rhysjones/phi-2-orange")
tokenizer = AutoTokenizer.from_pretrained("rhysjones/phi-2-orange")

# Optimize with IPEX and set model to evaluation mode
model = ipex.llm.optimize(model, dtype=torch.bfloat16)  # Apply BF16 optimization
model = model.eval()

# Device handling (ensure IPEX device is available)
device = torch.device("cpu")
model = model.to(device)

# User prompt
user_prompt = input("Enter your question: ")

# Construct simplified ChatML prompt
prompt = f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n<|im_start|>assistant"

# Generate response with truncated max length and lower temperature
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(device)  # Move inputs to device
outputs = model.generate(
    **inputs,
    max_length=200,
    do_sample=True,
    top_p=0.92,
    temperature=0.7
)

# Decode and truncate based on full stops
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
stop_indices = [i for i, char in enumerate(text) if char == "."]
truncated_text = text[:stop_indices[2] + 1]

# Remove redundant system prompts
truncated_text = truncated_text.split("<|im_end|>")[1].strip()

print(truncated_text)
