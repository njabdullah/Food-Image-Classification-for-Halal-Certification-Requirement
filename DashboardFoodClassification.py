import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import pandas as pd
import time

# Set page config as the first Streamlit command
st.set_page_config(page_title="Klasifikasi Sertifikasi Halal", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f9fafb; }
    .stButton>button { 
        background-color: #10b981; 
        color: white; 
        border-radius: 8px; 
        padding: 10px 20px; 
        font-weight: 500; 
        border: none; 
    }
    .stButton>button:hover { 
        background-color: #059669; 
    }
    .result-card { 
        background-color: white; 
        padding: 20px; 
        border-radius: 10px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        margin-bottom: 20px; 
    }
    .title { 
        color: #1f2937; 
        font-size: 2.5rem; 
        font-weight: 700; 
        margin-bottom: 1rem; 
    }
    .subtitle { 
        color: #4b5563; 
        font-size: 1.2rem; 
        margin-bottom: 2rem; 
    }
    </style>
""", unsafe_allow_html=True)

# Load all models
@st.cache_resource
def load_models():
    model_paths = {
        'MobileNet Fine-Tuned': 'Model/FineTuned/halal_certification_mobilenetv2.h5',
        'MobileNet': 'Model/Base/halal_certification_mobilenetv2.h5',
        'Inception': 'Model/Base/halal_certification_inceptionv3.h5'
    }
    return {name: load_model(path) for name, path in model_paths.items()}

def preprocess_image(uploaded_file, target_size=(128, 128)):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img

def predict_with_all_models(models, img_array):
    results = {}
    for model_name, model in models.items():
        pred = model.predict(img_array)[0][0]
        label = "Perlu Sertifikasi Kehalalan" if pred < 0.5 else "Tidak Perlu Sertifikasi Kehalalan"
        confidence = (1 - pred) * 100 if pred < 0.5 else pred * 100
        results[model_name] = (label, confidence)
    return results

# Streamlit UI
st.markdown('<div class="title">🕌 Klasifikasi Sertifikasi Halal</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Unggah gambar produk untuk menentukan apakah memerlukan sertifikasi halal menggunakan model AI canggih.</div>', unsafe_allow_html=True)

# File uploader
with st.container():
    uploaded_file = st.file_uploader("Pilih gambar produk", type=["jpg", "jpeg", "png"], help="Unggah file gambar (JPG, JPEG, atau PNG)")

if uploaded_file:
    with st.spinner("Memproses gambar..."):
        # Simulate processing time
        time.sleep(1)
        img_array, original_img = preprocess_image(uploaded_file)
        models = load_models()
        results = predict_with_all_models(models, img_array)

    # Layout with columns
    left_col, right_col = st.columns([1, 2], gap="large")
    
    with left_col:
        st.image(original_img, caption="Gambar Produk", use_container_width=True, clamp=True)
    
    with right_col:
        st.markdown("### 📊 Hasil Klasifikasi")
        # Create a DataFrame for results
        df_results = pd.DataFrame(
            [(model, label, f"{confidence:.2f}%") for model, (label, confidence) in results.items()],
            columns=["Model", "Prediksi", "Confidence"]
        )
        # Style the DataFrame
        st.dataframe(
            df_results.style.set_properties(**{
                'background-color': 'white',
                'color': '#1f2937',
                'border-color': '#e5e7eb',
                'text-align': 'left',
                'font-size': '20px'
            }).set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#10b981'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'left'), ('padding', '10px')]},
                {'selector': 'td', 'props': [('padding', '10px')]}
            ]),
            use_container_width=True
        )

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #6b7280;">Dibuat dengan Streamlit dan TensorFlow | © 2025</div>', unsafe_allow_html=True)