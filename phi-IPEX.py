import torch
import intel_extension_for_pytorch as ipex
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

#ipex.enable_onednn_fusion(True) 

device = torch.device("cpu")
start_time = time.time()

# Load Ipex-optimized model and tokenizer (if available)
model = ipex.optimize(AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, trust_remote_code=True)).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

user_prompt = input("Enter your prompt: ")

inputs = tokenizer(user_prompt, return_tensors="pt", return_attention_mask=False)

# Move input tensors to the Ipex-optimized device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Use Ipex-optimized model for inference
outputs = model.generate(
    **inputs,
    max_length=200,
    do_sample=True,
    top_p=0.92,
    temperature=0.85
)

end_time = time.time()
print(f"Inference Time with Ipex: {end_time - start_time} seconds")

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
