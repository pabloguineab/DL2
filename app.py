import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

def load_frozen_graph(path):
    with tf.io.gfile.GFile(path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        
    return graph

# Load the trained model
graph = load_frozen_graph('models/frozen_inference_graph.pb')

# Use the graph
with tf.compat.v1.Session(graph=graph) as sess:
    input_image = graph.get_tensor_by_name('input_image:0')  # Replace with the actual names
    output = graph.get_tensor_by_name('output:0')  # Replace with the actual names

def predict(image):
    # Convert the image to numpy array
    image_array = np.array(image)

    # Preprocess the image
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    image_array = cv2.resize(image_array, (299, 299)) # size should be adjusted according to your model's input shape
    image_array = np.expand_dims(image_array, axis=0)

    # Perform inference
    predictions = sess.run(output, feed_dict={input_image: image_array})

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
