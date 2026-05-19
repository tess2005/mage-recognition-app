import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# 1. Load the "Brain" (The AI Model) - Cache it so it only loads once
@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

model = load_model()

st.title("Elite Image Recognition Terminal 🧠")
st.write("Upload an image, and the AI will identify the object!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocessing the image to match what the model expects
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Make prediction
    predictions = model.predict(img_array)
    decoded = decode_predictions(predictions, top=3)[0] # Get top 3 guesses
    
    st.write("### AI Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded):
        st.write(f"{i+1}: **{label}** ({score*100:.2f}% confidence)")