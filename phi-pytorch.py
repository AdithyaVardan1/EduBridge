import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set device to CPU as you're running without IPEX optimizations
device = torch.device("cpu")
start_time = time.time()

# Load the model and tokenizer, ensuring they're set to use the CPU
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

user_prompt = input("Enter your prompt: ")

# Tokenize the user input
inputs = tokenizer(user_prompt, return_tensors="pt", return_attention_mask=False)

# Move input tensors to the same device as the model (CPU)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate the model's output based on the inputs
outputs = model.generate(
    **inputs,
    max_length=200,
    do_sample=True,
    top_p=0.92,
    temperature=0.85
)

# Measure the inference time
end_time = time.time()
print(f"Inference Time: {end_time - start_time} seconds")

# Decode the generated tokens to a string and print
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
