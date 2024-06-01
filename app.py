import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
import cv2

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\uppload'
app.config['MAX_CONTENT_PATH'] = 1024 * 1024  # 1MB limit

# Load your trained model
MODEL_PATH = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Models\\best_model.h5'
model = load_model(MODEL_PATH)

def preprocess_image(image_path):
    """Preprocess the image for model prediction"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image and predict
            image = preprocess_image(file_path)
            prediction = model.predict(image)
            is_ai = prediction[0][0] > 0.5

            result = "AI Generated" if is_ai else "Real Image"
            flash(result)
            return redirect(url_for('index'))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
