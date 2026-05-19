import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import requests

# 1. Load the "Brain"
@st.cache_resource
def load_model():
    model = models.resnet18(weights='DEFAULT')
    model.eval()
    return model

model = load_model()

# 2. Get the labels
def get_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    return response.text.splitlines()

labels = get_labels()

# 3. Setup Image processing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.title("AI Vision System 🧠")
uploaded_file = st.file_uploader("Upload image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Analyzing...', use_column_width=True)
    
    # Process and Predict
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        
    # Get the name of the object
    predicted_label = labels[predicted_idx.item()]
    
    st.write(f"### AI Analysis Result:")
    st.write(f"The AI thinks this is: **{predicted_label}**")