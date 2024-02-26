import torch
import intel_extension_for_pytorch as ipex
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("rhysjones/phi-2-orange")
tokenizer = AutoTokenizer.from_pretrained("rhysjones/phi-2-orange")

# Optimize with IPEX and set model to evaluation mode
model = ipex.llm.optimize(model, dtype=torch.bfloat16)  # Apply BF16 optimization
model.eval()  # Set the model to evaluation mode

# Device handling (set to CPU explicitly)
device = torch.device("cpu")
model.to(device)

# User prompt
user_prompt = input("Enter your question: ")

# Construct simplified ChatML prompt with consistent user prefix
prompt = f"user: {user_prompt}\n\nassistant:"

# Generate response with adjusted parameters for better control
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(device)
outputs = model.generate(
    **inputs,
    max_length=400,  # Adjust max length if necessary
    do_sample=True,
    top_p=0.92,
    temperature=0.7,
    num_return_sequences=1,  # Consider generating multiple sequences if needed
    pad_token_id=tokenizer.eos_token_id
)

# Decode and improve truncation based on full stops
text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Truncate text after the third full stop if there are enough, else return full text
full_stops = [i for i, char in enumerate(text) if char == "."]
if len(full_stops) >= 3:
    truncated_text = text[:full_stops[2] + 1]
else:
    truncated_text = text

print(truncated_text)
