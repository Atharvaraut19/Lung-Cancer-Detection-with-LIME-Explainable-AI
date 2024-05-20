from flask import Flask, render_template, request
from lime import lime_image
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('efficientnet_model.h5')

# Initialize the LIME explainer
explainer_lime = lime_image.LimeImageExplainer()


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
    
    # Darken the boundaries color
    explanation_img = mark_boundaries(temp / 2 + 0.5, mask, color=(0, 0, 0))
    
    return explanation_img

if __name__ == '__main__':
    app.run(debug=True)
