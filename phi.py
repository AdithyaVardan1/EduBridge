import torch
import intel_extension_for_pytorch as ipex
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cpu")
start_time = time.time()

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, trust_remote_code=True).to(ipex.DEVICE)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

user_prompt = input("Enter your prompt: ")

inputs = tokenizer(user_prompt, return_tensors="pt", return_attention_mask=False)
inputs = inputs.data.to(ipex.DEVICE)

outputs = model.generate(
    **inputs,
    max_length=500,
    do_sample=True,
    top_p=0.92,
    top_k=50,  # Adjusted parameter
    temperature=0.75  # Slightly lower temperature for more focus
)

end_time = time.time()
print(f"Inference Time: {end_time - start_time} seconds")

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
