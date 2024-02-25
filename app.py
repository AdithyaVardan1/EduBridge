import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify

# Load LLM model and tokenizer (replace with your chosen LLM)
model_name = "rhysjones/phi-2-orange"
device = torch.device("cpu")  # Assuming no GPU
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create Flask app
app = Flask(__name__)

# API endpoint to receive and respond to messages
@app.route("/message", methods=["POST"])
def handle_message():
    # Retrieve message from request
    message = request.json["message"]

    # Generate response using the LLM
    inputs = tokenizer(message, return_tensors="pt", return_attention_mask=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_length=200, do_sample=True, top_p=0.92, temperature=0.7)

    # Decode and truncate response (adjust truncation logic as needed)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    truncated_text = text[:200]  # Truncate to 200 characters initially
    stop_indices = [i for i, char in enumerate(text) if char == "."]
    if len(stop_indices) > 2:
        truncated_text = text[:stop_indices[2] + 1]  # Truncate at the third full stop if possible

    # Return the response
    return jsonify({"response": truncated_text})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")  # Bind to all interfaces for public access
