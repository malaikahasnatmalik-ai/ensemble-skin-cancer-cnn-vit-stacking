import streamlit as st
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="Skin Cancer Detection", page_icon="🔬", layout="centered")

st.title("🔬 Skin Cancer Classification")
st.markdown("**Developed by Malaika Aziz Ahmed | The University of Agriculture, Peshawar**")
st.markdown("---")

# Model loading with caching
MODEL_NAME = "c:/Users/lcmal/OneDrive/Desktop/Skin Cancer Project/skincancer.py"

@st.cache_resource
def load_skin_model():
    if os.path.exists(MODEL_NAME):
        return load_model(MODEL_NAME)
    elif os.path.exists("Meta_Learner_Model (2).h5"):
        return load_model("Meta_Learner_Model (2).h5")
    else:
        return None

model = load_skin_model()

if model is None:
    st.error(f"Model file not found! Please make sure {MODEL_NAME} is in the same folder.")
else:
    st.success("Model loaded successfully!")

# Class labels
class_labels = ["Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma", "Actinic Keratosis", "Benign Keratosis", "Dermatofibroma", "Vascular Lesion"]
class_desc = ["Dangerous melanoma cancer", "Common treatable skin cancer", "Second most common skin cancer", "Precancerous sun damage", "Benign harmless growth", "Benign firm tumor", "Benign blood vessel growth"]

uploaded_file = st.file_uploader("Upload Skin Lesion Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300, caption="Uploaded Image")

    # Preprocess
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Predict Now", use_container_width=True):
        with st.spinner("Analyzing..."):
            pred = model.predict(img_array)
            idx = np.argmax(pred)
            conf = np.max(pred) * 100

            st.balloons()
            st.markdown(f"### Result: **{class_labels[idx]}**")
            st.progress(int(conf))
            st.write(f"**Confidence: {conf:.2f}%**")
            st.info(f"{class_desc[idx]}")
            st.warning("Note: This is an AI prediction for academic purpose. Please consult a dermatologist.")

st.markdown("---")
st.caption("FYP Project - NCAI Demo 2026")





# --- BACKGROUND CODE ADDED HERE ---
def add_bg_from_local(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
           .stApp {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
           .stApp > header {{
                background-color: transparent;
            }}
            div[data-testid="stVerticalBlock"] {{
                background-color: rgba(255,255,255,0.85);
                padding: 20px;
                border-radius: 15px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

