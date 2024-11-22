from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch
import os
import base64
from io import BytesIO
import string
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from path_config import MODEL_PATHS
 
app = Flask(__name__)
device="cuda"
# Load models dynamically using MODEL_PATHS
model_cache = {}
def load_model(model_name):
    if model_name in model_cache:
        return model_cache[model_name], None

    model_path = os.path.join('models', f'{model_name}')
    print(f"Loading model from: {model_path}")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_path, revision="fp16", torch_dtype=torch.float16)
        pipe.to(device)
        model_cache[model_name] = pipe
        return pipe, None
    except Exception as e:
        return None, str(e)

# for model_name, model_path in MODEL_PATHS.items():
#     pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
#     pipe = pipe.to("cuda")
#     models[model_name] = pipe
 
def generate_random_string(length=5):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))
 
def generate_unique_filename(prompt):
    # Take first 10 characters of the prompt
    prompt_prefix = prompt[:10]
   
    # Generate a random string for uniqueness
    unique_string = generate_random_string()
   
    # Concatenate prompt prefix and unique string
    filename = f"{prompt_prefix}_{unique_string}.png"
    return filename
 
@app.route('/a_generate_image', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get('prompt')
    model_name = data.get('model_name')

    if not prompt:
        return jsonify({'error': 'Prompt not provided'}), 400

    if not model_name:
        return jsonify({'error': 'Model name not provided'}), 400

    # Load model if not in cache
    if model_name not in model_cache:
        model, error = load_model(model_name)
        if error:
            return jsonify({'error': f"Failed to load model: {error}"}), 500
    else:
        model = model_cache[model_name]

    # Generate image
    image = model(prompt).images[0]
 
    # Generate a unique filename based on the prompt
    filename = generate_unique_filename(prompt)
   
    # Define the directory path
    directory_path = "result"
 
    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
 
    # Save the generated image
    output_path = os.path.join(directory_path, filename)
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
 
    # Convert image to base64
    with open(output_path, "rb") as f:
        image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode()
 
    return jsonify({'image_base64': image_base64})
 
if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5010)