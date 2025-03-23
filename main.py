import json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Google Drive File ID for the model
FILE_ID = "169KurWfpPT0m3eoPKG0cHtxYDc1GvJiO"

# Function to download the model from Google Drive
def download_model_from_drive(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open("plant_disease_prediction_model.h5", "wb") as f:
            f.write(response.content)
        return "plant_disease_prediction_model.h5"
    else:
        st.error("Failed to download model.")
        return None

# Load model from Google Drive
model_path = download_model_from_drive(FILE_ID)
if model_path:
    model = tf.keras.models.load_model(model_path)

# Load class indices from JSON file (Ensure this file is in your project directory)
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Function to preprocess image
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)  # Resize image
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict image class
def predict_image_class(image):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions)  # Get class index
    return class_indices[str(predicted_class_index)]  # Map index to class name

# Streamlit UI
st.title("🌿 Plant Disease Classifier 🌿")

st.markdown("""
### Instructions:
1. Upload a plant leaf image.
2. Click "Classify" to predict the disease.
""")

# File uploader for image input
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        if st.button("Classify"):
            with st.spinner("Classifying..."):
                prediction = predict_image_class(image)
            st.success(f"**Prediction:** {prediction}")
