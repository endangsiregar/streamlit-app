import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import json

# Load model and descriptions
MODEL_PATH = "./model/best_efficientnet_model.keras"
DESCRIPTIONS_PATH = "./descriptions.json"

# Load the trained model
model = load_model(MODEL_PATH)

# Load ulos descriptions
with open(DESCRIPTIONS_PATH, "r") as file:
    ulos_descriptions = json.load(file)

# Class names
CLASS_NAMES = list(ulos_descriptions.keys())

# Streamlit App
st.title("Klasifikasi dan Deskripsi Ulos")
st.write("Unggah gambar ulos Anda untuk mengetahui jenisnya dan mendapatkan informasi tentang ulos tersebut.")

# File uploader
uploaded_file = st.file_uploader("Pilih gambar ulos", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Anda unggah", use_column_width=True)

    # Preprocess the image
    image = image.resize((299, 299))
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = predictions[0][predicted_index]

    # Display results
    st.subheader(f"Jenis Ulos: {predicted_class}")
    st.write(f"Tingkat Kepercayaan: {confidence * 100:.2f}%")
    st.write("### Deskripsi:")
    st.write(ulos_descriptions[predicted_class])
