import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Configuración
IMG_HEIGHT = 224
IMG_WIDTH = 224

# ==============================
# FUNCIONES DE CARGA
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mejor_modelo.h5")

@st.cache_resource
def load_classes():
    with open("clases.json", "r") as f:
        return json.load(f)

# Cargar modelo y clases
model = load_model()
class_names = load_classes()

# ==============================
# FUNCIÓN DE PREDICCIÓN
# ==============================
def predict_image(image: Image.Image):
    img_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# ==============================
# INTERFAZ
# ==============================
st.set_page_config(page_title="Clasificador de Imágenes", layout="centered")
st.title("🤖 Clasificador de Imágenes con MobileNetV2")
st.write("Usa este modelo entrenado para clasificar imágenes desde tu PC o tu cámara.")

# Menú lateral
modo = st.sidebar.radio("Selecciona el modo", ["📂 Subir imagen", "📷 Usar cámara"])

# ==============================
# MODO 1: SUBIR IMAGEN
# ==============================
if modo == "📂 Subir imagen":
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagen subida", use_column_width=True)
        
        predicted_class, confidence = predict_image(image)
        st.markdown(f"**Predicción:** {predicted_class}")
        st.markdown(f"**Confianza:** {confidence:.2f}%")

# ==============================
# MODO 2: USAR CÁMARA
# ==============================
elif modo == "📷 Usar cámara":
    camera_image = st.camera_input("Toma una foto")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="Imagen capturada", use_column_width=True)
        
        predicted_class, confidence = predict_image(image)
        st.markdown(f"**Predicción:** {predicted_class}")
        st.markdown(f"**Confianza:** {confidence:.2f}%")
