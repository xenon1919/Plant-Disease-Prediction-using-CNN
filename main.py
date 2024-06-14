import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Get the current working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to the model and class indices files
model_path = os.path.join(working_dir, "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the class names
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# Function to load and preprocess the image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit app
st.title('🌿 Plant Disease Classifier 🌿')

st.markdown("""
    ### Instructions:
    1. Upload an image of a plant leaf.
    2. Click the "Classify" button to predict the disease.
""")

# File uploader for the image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.markdown(f"**Image details:**\n- Dimensions: {image.size[0]} x {image.size[1]}\n- Format: {image.format}")

    with col2:
        if st.button('Classify'):
            with st.spinner('Classifying...'):
                prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'**Prediction:** {str(prediction)}')
