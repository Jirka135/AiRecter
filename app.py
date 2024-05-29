import os
import time
import threading
from flask import Flask, render_template, request, redirect, url_for, flash, Response
from flask_sse import sse
from redis import Redis, ConnectionError
from train import train_ai
from preprocess import preprocess_images, load_preprocessed_data
from image_generation import call_txt2img_api, read_prompts_from_file

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config["REDIS_URL"] = "redis://192.168.0.125:6379"
app.register_blueprint(sse, url_prefix='/stream')

def check_redis_connection():
    try:
        redis = Redis.from_url(app.config["REDIS_URL"])
        redis.ping()
        return True
    except ConnectionError:
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    if not check_redis_connection():
        flash('Redis server is not running. Please start the Redis server and try again.', 'danger')
        return redirect(url_for('home'))

    threading.Thread(target=train_ai_with_feedback).start()
    flash('Training started', 'success')
    return redirect(url_for('home'))

@app.route('/preprocess', methods=['POST'])
def preprocess():
    if not check_redis_connection():
        flash('Redis server is not running. Please start the Redis server and try again.', 'danger')
        return redirect(url_for('home'))

    ai_generated_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Images\\Fake'
    real_images_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Images\\Real'
    train_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Images\\Train'
    test_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Images\\Test'

    threading.Thread(target=preprocess_images_with_feedback, args=(ai_generated_dir, real_images_dir, train_dir, test_dir)).start()
    flash('Preprocessing started', 'success')
    return redirect(url_for('home'))

@app.route('/stream')
def stream():
    return Response(sse.stream(), content_type='text/event-stream')

def preprocess_images_with_feedback(ai_generated_dir, real_images_dir, train_dir, test_dir):
    with app.app_context():
        preprocess_images(ai_generated_dir, real_images_dir, train_dir, test_dir, test_size=0.2, sse=sse)

def train_ai_with_feedback():
    with app.app_context():
        train_ai(sse=sse)

if __name__ == '__main__':
    app.run(debug=True)
