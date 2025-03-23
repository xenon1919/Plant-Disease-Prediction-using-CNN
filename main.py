import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Load the pre-trained model
model = tf.keras.models.load_model("plant_disease_prediction_model.h5")

# Load the class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Function to load and preprocess the image
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict the class of an image
def predict_image_class(image):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions)
    return class_indices[str(predicted_class_index)]

# Streamlit app UI
st.title('🌿 Plant Disease Classifier 🌿')

st.markdown("""
### Instructions:
1. Upload an image of a plant leaf.
2. Click the "Classify" button to predict the disease.
""")

# File uploader for the image
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
