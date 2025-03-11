from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from transformers import pipeline
from PIL import Image

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the ML model
pipe = pipeline("image-classification", model="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/crop-disease')
def crop_disease():
    return render_template('crop_disease.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template("error.html", message="No file part")

    file = request.files['image']
    if file.filename == '':
        return render_template("error.html", message="No selected file")

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the image with ML model
        image = Image.open(file_path)
        results = pipe(image)
        results = sorted(results, key=lambda x: x['score'], reverse=True)  # Sort by confidence

        return render_template("result.html", results=results, image_path=file_path)


if __name__ == '__main__':
    app.run(debug=True)
