import os
import time
import threading
import subprocess
import redis
from flask import Flask, render_template, redirect, url_for, jsonify, send_from_directory
from flask_sse import sse
from preprocess import preprocess_images
from train import train_ai
from image_generation import call_txt2img_api, read_prompts_from_file, update_eta
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import plot_model

app = Flask(__name__)
app.config["REDIS_URL"] = "redis://192.168.0.125:6379"
app.register_blueprint(sse, url_prefix='/stream')

# Define paths for logging and model checkpoints
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

def is_redis_available():
    try:
        r = redis.StrictRedis.from_url(app.config["REDIS_URL"])
        r.ping()
        return True
    except redis.ConnectionError:
        return False

@app.route('/')
def index():
    redis_status = is_redis_available()
    return render_template('index.html', redis_status=redis_status)

@app.route('/check_redis')
def check_redis():
    redis_status = is_redis_available()
    return jsonify(redis_status=redis_status)

@app.route('/preprocess', methods=['POST'])
def preprocess():
    def preprocess_images_with_feedback():
        with app.app_context():
            ai_generated_dir = 'D:\\AIimages\\Fake'
            real_images_dir = 'D:\\AIimages\\Real'
            train_dir = 'D:\\AIimages\\Train'
            test_dir = 'D:\\AIimages\\Test'
            preprocess_images(ai_generated_dir, real_images_dir, train_dir, test_dir, test_size=0.2, sse=sse)

    threading.Thread(target=preprocess_images_with_feedback).start()
    return redirect(url_for('index'))

@app.route('/train', methods=['POST'])
def train():
    threading.Thread(target=lambda: train_ai(sse)).start()
    return redirect(url_for('index'))

@app.route('/generate', methods=['POST'])
def generate():
    def generate_images():
        with app.app_context():
            prompt_file = 'prompts.txt'
            prompts = read_prompts_from_file(prompt_file)
            if not prompts:
                print("No prompts found in the file. Please add prompts to 'prompts.txt' and run the script again.")
            else:
                negative_prompt = "low quality, blurry, bad anatomy, disfigured, deformed, extra limbs, extra fingers, extra arms, extra legs, poorly drawn, poorly rendered, bad proportions, unnatural body, unnatural lighting, watermark, text, logo, nsfw, low resolution, grainy, overexposed, underexposed, distorted, cropped, jpeg artifacts, watermark, cartoon, sketch, 3d render, anime, unrealistic, bad art, ugly, messy, cluttered, low detail, noise, overprocessed"
                total_prompts = len(prompts)
                start_time = time.time()
                processed_prompts = [0]
                stop_event = threading.Event()

                # Start the ETA update thread
                eta_thread = threading.Thread(target=update_eta, args=(total_prompts, start_time, processed_prompts, stop_event))
                eta_thread.start()

                try:
                    for prompt in prompts:
                        call_txt2img_api(prompt, negative_prompt)
                        processed_prompts[0] += 1  # Increment the count of processed prompts
                finally:
                    stop_event.set()
                    eta_thread.join()
                print("\nAll prompts processed.")

    threading.Thread(target=generate_images).start()
    return redirect(url_for('index'))

@app.route('/tensorboard')
def launch_tensorboard():
    subprocess.Popen(['tensorboard', '--logdir=logs', '--port=6006'])
    return redirect("http://localhost:6006", code=302)

@app.route('/model_architecture')
def model_architecture():
    try:
        model_architecture_path = os.path.join('model_checkpoints', 'model_architecture.png')
        if not os.path.exists(model_architecture_path):
            raise FileNotFoundError(f"Model architecture image not found at path: {model_architecture_path}")
        return send_from_directory('model_checkpoints', 'model_architecture.png')
    except Exception as e:
        return str(e), 404

@app.route('/weights')
def visualize_weights():
    try:
        model_path = os.path.join('model_checkpoints', 'ai_image_recognition_model_final.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at path: {model_path}")

        model = load_model(model_path)
        layer = model.layers[1]  # Adjusted to use the correct layer index
        weights = layer.get_weights()[0]

        fig, ax = plt.subplots()
        cax = ax.matshow(weights, cmap='viridis')
        plt.title('Layer 1 Weights')
        plt.colorbar(cax)

        img_path = 'static/weights.png'
        os.makedirs('static', exist_ok=True)
        plt.savefig(img_path)
        return send_from_directory('static', 'weights.png')
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=False)
