import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import io


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.h5")
    return model

best_model = load_model()


class_names = [
    "Bear", "Bird", "Cat", "Cow", "Deer",
    "Dog", "Dolphin", "Elephant", "Giraffe", "Horse",
    "Kangaroo", "Lion", "Panda", "Tiger", "Zebra"
]


def predict(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = best_model.predict(img_array)
    pred_index = np.argmax(preds)
    pred_class = class_names[pred_index]
    confidence = np.max(preds) * 100
    return pred_class, confidence


st.set_page_config(page_title="🐾 Animal Classifier", layout="centered")

st.title("🐾 Animal Classification App")
st.markdown("Upload an image of an animal and let the model predict which one it is!")

uploaded_file = st.file_uploader("Upload an animal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file)

    # Show uploaded image
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Predict button
    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            pred_class, confidence = predict(img)

        st.success(f"### ✅ Prediction: **{pred_class}**")
        st.info(f"📊 Confidence: {confidence:.2f}%")
