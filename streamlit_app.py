import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision import transforms
from skimage.feature import hog
from skimage.color import rgb2gray
import joblib

st.title("🔍 Hotdog or Not? | Multi-Model Demo")

MODEL_OPTIONS = [
    "HOG + Logistic Regression (v1)",
    "MobileNetV2 (v2)",
    "ResNet18 (v3/v4)"
]

@st.cache_resource
def load_logreg():
    import os
    return joblib.load(os.path.join("models", "logreg_model.pkl"))

@st.cache_resource
def load_mobilenet():
    model = torchvision.models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
    state = torch.load("models/mobilenetv2_state.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

@st.cache_resource
def load_resnet():
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    state = torch.load("models/best_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

model_choice = st.selectbox("Select model version", MODEL_OPTIONS)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    pred = None

    if model_choice == "HOG + Logistic Regression (v1)":
        st.markdown("🧠 Preprocessing with `HOG` features")
        gray = rgb2gray(np.array(image.resize((128, 128))))
        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        model = load_logreg()
        pred = model.predict([features])[0]

    else:
        st.markdown("🧠 Preprocessing with `Torchvision Transforms`")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        tensor = transform(image).unsqueeze(0)
        if model_choice == "MobileNetV2 (v2)":
            model = load_mobilenet()
        else:
            model = load_resnet()
        with torch.no_grad():
            output = model(tensor)
            pred = torch.argmax(output, dim=1).item()

    if pred is not None:
        # For all models: assume 0 = HOTDOG, 1 = NOT HOTDOG
        label = "🌭 HOTDOG" if pred == 0 else "🚫 NOT HOTDOG"
        st.success(f"Prediction: {label}")

