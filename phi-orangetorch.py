import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Disable Ipex for CPU (assuming no GPU)
# ipex.enable_onednn_fusion(True)

device = torch.device("cpu")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("rhysjones/phi-2-orange").to(device)
tokenizer = AutoTokenizer.from_pretrained("rhysjones/phi-2-orange")

# User prompt
user_prompt = input("Enter your question: ")

# Construct simplified ChatML prompt
prompt = f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n<|im_start|>assistant"

# Generate response with adjusted max length
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
inputs = {k: v.to(device) for k, v in inputs.items()}
outputs = model.generate(
    **inputs,
    max_length=300,  # Assuming 2-3 paragraphs
    do_sample=True,
    top_p=0.92,
    temperature=0.75
)

# Decode and format response
text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Remove redundant system prompts
text = text.split("<|im_end|>")[1].strip()  # Keep only assistant's response

# Optionally add a concluding sentence for summary
# text += "\nIn summary, text embeddings..."  # Modify as needed

print(text)
