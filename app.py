from flask import Flask, request, jsonify
import torch
import intel_extension_for_pytorch as ipex
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Initialize the model and tokenizer when the app starts
model = AutoModelForCausalLM.from_pretrained("rhysjones/phi-2-orange")
tokenizer = AutoTokenizer.from_pretrained("rhysjones/phi-2-orange")
model = ipex.llm.optimize(model, dtype=torch.bfloat16)  # Apply BF16 optimization
model = model.eval()
model.to(torch.device("cpu"))

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    user_prompt = data.get("prompt", "")
    
    # Ensure the prompt is not empty
    if not user_prompt:
        return jsonify({"error": "Empty prompt provided."}), 400
    
    prompt = f"user: {user_prompt}\n\nassistant"
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(torch.device("cpu"))
    outputs = model.generate(
        **inputs,
        max_length=200,
        do_sample=True,
        top_p=0.92,
        temperature=0.7
    )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    stop_indices = [i for i, char in enumerate(text) if char == "."]
    
    # Handle case with fewer than 3 full stops
    if len(stop_indices) >= 3:
        truncated_text = text[:stop_indices[2] + 1]
    else:
        truncated_text = text
    
    return jsonify({"response": truncated_text})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
