import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Define the classes
classes = ["No Cancer", "Adenocarcinoma", "Squamous Cell Carcinoma"]

# Load the pre-trained model
model = load_model('efficientnet_model.h5')

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalization
    return img_array

# Function to perform classification on a single image
def classify_image(image_path):
    # Load and preprocess the image
    img_array = load_and_preprocess_image(image_path)
    
    # Perform prediction
    prediction = model.predict(img_array)
    
    # Get the class label with highest probability
    predicted_class_index = np.argmax(prediction)
    predicted_class = classes[predicted_class_index]
    
    # Get the probability of the predicted class
    confidence = prediction[0][predicted_class_index]
    
    return predicted_class, confidence

# Path to the image you want to classify
image_path = 'lungaca12.jpeg'

# Perform classification
predicted_class, confidence = classify_image(image_path)

# Print the result
print("Predicted Class:", predicted_class)
print("Confidence:",confidence)