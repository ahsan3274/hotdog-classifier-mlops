import streamlit as st
from PIL import Image
import numpy as np
import requests
import torch
from torchvision import transforms
from skimage.feature import hog
from skimage.color import rgb2gray

# Model endpoints
MODEL_ENDPOINTS = {
    "HOG + Logistic Regression (v1)": "http://127.0.0.1:5002/invocations",
    "MobileNetV2 (v2)": "http://127.0.0.1:5003/invocations",
    "ResNet18 (v3/v4)": "http://127.0.0.1:5001/invocations"
}

# UI
st.title("🔍 Hotdog or Not? | Multi-Model Demo")

model_choice = st.selectbox("Select model version", list(MODEL_ENDPOINTS.keys()))
endpoint_url = MODEL_ENDPOINTS[model_choice]

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if model_choice == "HOG + Logistic Regression (v1)":
        st.markdown("🧠 Preprocessing with `HOG` features")
        gray = rgb2gray(np.array(image.resize((64, 64))))
        features = hog(gray,
                       orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       block_norm='L2-Hys')
        input_data = [features.tolist()]
    else:
        st.markdown("🧠 Preprocessing with `Torchvision Transforms`")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        tensor = transform(image).unsqueeze(0)
        input_data = tensor.view(1, -1).cpu().numpy().astype(np.float32).tolist()

    payload = {"instances": input_data}

    if st.button("🔍 Predict"):
        with st.spinner("Sending to model..."):
            try:
                response = requests.post(endpoint_url, json=payload)
                if response.status_code == 200:
                    prediction = response.json()["predictions"][0]
                    if isinstance(prediction, list):
                        prediction = int(np.argmax(prediction))
                    else:
                        prediction = int(prediction)

                    # For all models: assume 0 = HOTDOG, 1 = NOT HOTDOG
                    label = "🌭 HOTDOG" if prediction == 0 else "🚫 NOT HOTDOG"
                    st.success(f"Prediction: {label}")
                else:
                    st.error(f"Server error {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"❌ Error: {e}")
