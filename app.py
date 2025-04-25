import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load your trained model
model = load_model('my_model.h5')

# Define class labels (must match your training data)
# Example: ['cats', 'dogs']
class_labels = ['hari', 'khusi', 'pushpa', 'ram', 'sudh']  # replace with your actual class names

st.title("Image Classification with CNN")
st.write("Upload an image to classify it.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image_resized = image.resize((128, 128))
    image_array = img_to_array(image_resized) / 255.0  # normalize
    image_batch = np.expand_dims(image_array, axis=0)  # add batch dimension

    # Make prediction
    predictions = model.predict(image_batch)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_index]
    confidence = predictions[0][predicted_index]

    # Show prediction
    st.write(f"Prediction: **{predicted_label}**")
    st.write(f"Confidence: {confidence:.2f}")

    # Optional: Show all class probabilities
    st.subheader("Class Probabilities")
    for i, label in enumerate(class_labels):
        st.write(f"{label}: {predictions[0][i]:.2f}")
