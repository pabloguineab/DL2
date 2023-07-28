import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
model = tf.saved_model.load('models/frozen_inference_graph.pb')

def predict(image):
    # Convert the image to numpy array
    image_array = np.array(image)

    # Preprocess the image
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    image_array = cv2.resize(image_array, (299, 299)) # size should be adjusted according to your model's input shape
    image_array = np.expand_dims(image_array, axis=0)

    # Perform inference
    predictions = model(image_array)

    # Postprocess predictions: here you might need to adapt the code to the specific output of your model
    card = np.argmax(predictions[0]) # index of the max confidence score
    suit = np.argmax(predictions[1]) # index of the max confidence score

    return card, suit

st.title("Poker Card Identifier")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Predicting...")

    # Call the prediction function
    card, suit = predict(image)

    st.write(f"The identified card is: {card}")
    st.write(f"The identified suit is: {suit}")