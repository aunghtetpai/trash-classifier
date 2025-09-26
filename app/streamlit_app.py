import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# -----------------------------
# Threading / performance fixes
# -----------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

IMG_SIZE = 224  # Teachable Machine default

# -----------------------------
# Load model & labels
# -----------------------------
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model("model/keras_model.h5")
    with open("model/labels.txt", "r") as f:
        labels = [line.strip() for line in f]
    return model, labels

# -----------------------------
# Predict function
# -----------------------------
def predict_image(model, labels, img):
    x = np.array(img).astype(np.float32)
    x = (x / 127.5) - 1.0  # MobileNet preprocessing
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x, verbose=0)[0]
    class_idx = int(np.argmax(pred))
    confidence = float(pred[class_idx])
    return labels[class_idx], confidence

# -----------------------------
# Main Streamlit app
# -----------------------------
def main():
    st.title("♻ Trash Classifier")

    uploaded_file = st.file_uploader("Upload a trash image", type=["jpg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        st.image(img, caption="Uploaded Image", use_container_width=True)

        model, labels = load_model_and_labels()
        pred_class, confidence = predict_image(model, labels, img)
        st.success(f"Prediction: **{pred_class}** — Confidence: {confidence*100:.1f}%")

if __name__ == "__main__":
    main()
