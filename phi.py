import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure everything runs on CPU, especially in a CPU-only environment
device = torch.device("cpu")

# Load model and tokenizer, explicitly setting to use CPU-compatible data types
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# Prepare inputs with tokenizer
inputs = tokenizer('''Explain closest pair of points approach.''', return_tensors="pt", return_attention_mask=False)

# Move input tensors to the same device as the model (CPU in this case)
inputs = inputs.to(device)

outputs = model.generate(
    **inputs,
    max_length=200,
    do_sample=True,  # Enable sampling to introduce randomness
    top_p=0.92,      # Use nucleus sampling
    temperature=0.85 # Adjust temperature to control randomness
)

# Decode and print the output text
text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(text)
