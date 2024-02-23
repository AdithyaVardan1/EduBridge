import torch
import intel_extension_for_pytorch as ipex
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set device to CPU as IPEX optimizations are for Intel CPUs
device = torch.device("cpu")
start_time  = time.time()
# Enable IPEX optimizations
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, trust_remote_code=True).to(ipex.DEVICE)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# Take prompt as input from the user
user_prompt = input("Enter your prompt: ")

# Prepare inputs with tokenizer
inputs = tokenizer(user_prompt, return_tensors="pt", return_attention_mask=False)

# Since the model is already on the IPEX device, inputs need to be converted to IPEX tensors as well
inputs = inputs.data.to(ipex.DEVICE)

outputs = model.generate(
    **inputs,
    max_length=500,
    do_sample=True,
    top_p=0.92,
    temperature=0.85
)

end_time = time.time()
print(f"Inference Time: {end_time - start_time} seconds")

# Decode and print the output text
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
