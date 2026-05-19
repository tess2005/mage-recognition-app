import streamlit as st
from PIL import Image

st.title("Simple Image Identifier 📸")
st.write("Upload an image, and I will display it for you!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Your Uploaded Image', use_column_width=True)
    st.write("Image successfully loaded and processed!")
    # We will build the advanced prediction logic once the app is stable!