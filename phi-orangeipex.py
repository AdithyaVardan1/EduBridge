import torch
import intel_extension_for_pytorch as ipex

from transformers import AutoModelForCausalLM, AutoTokenizer

ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)  # Step 2: Enable BF16 auto-mixed-precision

device = ipex.DEVICE  # Use IPEX device

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("rhysjones/phi-2-orange").to(device)
tokenizer = AutoTokenizer.from_pretrained("rhysjones/phi-2-orange")

# User prompt
user_prompt = input("Enter your question: ")

# Construct simplified ChatML prompt
prompt = f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n<|im_start|>assistant"

# Generate response with truncated max length and lower temperature
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to IPEX device
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
