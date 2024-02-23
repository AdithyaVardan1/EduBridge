import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure everything runs on CPU, especially in a CPU-only environment
device = torch.device("auto")

# Load model and tokenizer, explicitly setting to use CPU-compatible data types
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# Prepare inputs with tokenizer
inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

# Move input tensors to the same device as the model (CPU in this case)
inputs = inputs.to(device)

# Generate outputs
outputs = model.generate(**inputs, max_length=200)

# Decode and print the output text
text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(text)
