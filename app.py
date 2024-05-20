from flask import Flask, render_template, request
from lime import lime_image
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries
import tensorflow as tf
import base64
import os

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('efficientnet_model.h5')

# Initialize the LIME explainer
explainer_lime = lime_image.LimeImageExplainer()

# Define the upload folder
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize
    return image

def explain_with_lime(image):
    # Explain the prediction using LIME
    explanation_lime = explainer_lime.explain_instance(
        image, model.predict, top_labels=5, hide_color=None, num_samples=500
    )
    temp, mask = explanation_lime.get_image_and_mask(
        explanation_lime.top_labels[0], positive_only=True, num_features=5, hide_rest=False
    )
    
    # Change the boundaries color to red
    explanation_img = mark_boundaries(temp, mask, outline_color=(255, 0, 0))
    
    return explanation_img 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_image = request.files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
    uploaded_image.save(image_path)

    # Read the uploaded image
    image = Image.open(image_path)
    # Preprocess the image
    image = preprocess_image(image)
    # Perform prediction using the model
    prediction = model.predict(np.expand_dims(image, axis=0))
    
    # Explain the prediction using LIME
    explanation_lime = explain_with_lime(image)
    
    # Save the Lime explanation image
    explanation_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'lime_explanation.jpg')
    Image.fromarray((explanation_lime * 255).astype(np.uint8)).save(explanation_image_path)
    
    # Convert images to base64 format for HTML display
    uploaded_image_base64 = base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')
    explanation_lime_base64 = base64.b64encode(open(explanation_image_path, 'rb').read()).decode('utf-8')

    return render_template('result.html', 
                           prediction=prediction, 
                           uploaded_image_base64=uploaded_image_base64,
                           explanation_lime_base64=explanation_lime_base64)

if __name__ == '__main__':
    app.run(debug=True)
