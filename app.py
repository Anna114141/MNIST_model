from PIL import Image
import numpy as np
import streamlit as st
from tensorflow import keras

model = keras.models.load_model("mnist_model.h5")

st.title("Digit Recognition App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))
    image_np = np.array(image)

    image_np = 255 - image_np  # Invert colors
    image_np = image_np / 255.0
    image_flattened = image_np.reshape(1, 784)

    prediction = model.predict(image_flattened)
    predicted_digit = np.argmax(prediction)

    st.image(image, caption="Uploaded Digit Image", width=150)
    st.write(f"Predicted Digit: **{predicted_digit}**")
