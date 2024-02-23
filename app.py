from flask import Flask, request, jsonify
import torch
import intel_extension_for_pytorch as ipex
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Initialize the model and tokenizer on start-up
device = torch.device("cpu")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)


@app.route('/generate-text', methods=['POST'])
def generate_text():
    try:
        data = request.json
        user_prompt = data['prompt']
        
        inputs = tokenizer(user_prompt, return_tensors="pt", return_attention_mask=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_length=200,
            do_sample=True,
            top_p=0.92,
            temperature=0.85
        )
        end_time = time.time()

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({
            'response': text,
            'inference_time': f"{end_time - start_time} seconds"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
