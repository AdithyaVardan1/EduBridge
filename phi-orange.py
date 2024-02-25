
import torch
import intel_extension_for_pytorch as ipex
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

ipex.enable_onednn_fusion(True)  # Enable Ipex optimizations (if available)

device = torch.device("cpu")
start_time = time.time()

# Load model and tokenizer for rhysjones/phi-2-orange
model = AutoModelForCausalLM.from_pretrained("rhysjones/phi-2-orange", torch_dtype=torch.float32).to(device)
tokenizer = AutoTokenizer.from_pretrained("rhysjones/phi-2-orange")

# Construct ChatML prompt with system instruction
prompt = f"<|im_start|>system\nYou are a helpful assistant that provides answers in 2-3 paragraphs for theoretical questions or solutions for math questions or programs for program related questions.<|im_end|>\n<|im_start|>user\n{input('Enter your question: ')}\n<|im_end|>\n<|im_start|>assistant"

inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate response with appropriate length
outputs = model.generate(
    **inputs,
    max_length=600,  # Up to 3 paragraphs of 200 tokens each
    do_sample=True,
    top_p=0.92,
    temperature=0.85
)

end_time = time.time()
print(f"Inference Time with Ipex: {end_time - start_time:.2f} seconds")

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
