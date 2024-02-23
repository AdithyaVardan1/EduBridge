import torch
import intel_extension_for_pytorch as ipex
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set device to CPU as optimizations are for Intel CPUs
device = torch.device("cpu")
start_time = time.time()

# Enable IPEX optimizations by simply using .to(device) as you normally would
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

user_prompt = input("Enter your prompt: ")

inputs = tokenizer(user_prompt, return_tensors="pt", return_attention_mask=False)

# Move input tensors to the same device as the model
inputs = {k: v.to(device) for k, v in inputs.items()}

outputs = model.generate(
    **inputs,
    max_length=500,
    do_sample=True,
    top_p=0.92,
    temperature=0.85
)

end_time = time.time()
print(f"Inference Time: {end_time - start_time} seconds")

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
