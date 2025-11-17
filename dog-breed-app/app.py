# app.py – Flask frontend for dog breed prediction
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from classifier import DogBreedClassifier

# === Config ===
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
LABELS_PATH = '../dog-breed-identification/labels.csv'
IMAGES_DIR = '../dog-breed-identification/train/'
MODEL_PATH = 'best_model.h5'

# === Flask App ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Helper ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === Load Model ===
classifier = DogBreedClassifier(LABELS_PATH, IMAGES_DIR)
classifier.load_trained_model(MODEL_PATH)

# === Routes ===
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file is part of request
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return render_template('index.html', error="Invalid file format. Please upload a JPG/PNG image.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict breed
        top_prediction = classifier.predict_image(filepath, top_k=1)[0]
        return render_template('index.html', prediction=top_prediction, uploaded_image=filename)

    return render_template('index.html')

# === Run ===
if __name__ == '__main__':
    app.run(debug=True)