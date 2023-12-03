import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model
model =load_model('mnist_model_1.h5')

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert the image to grayscale
    image = image.convert('L')
    # Convert the image to a numpy array
    image = np.array(image)
    # Normalize the image
    image = image / 255.0
    # Reshape the image to match the input shape of the model
    image = np.reshape(image, (1, 28, 28, 1))
    return image

# Streamlit app
def main():
    st.title('Image Identifier')
    st.write('Upload an image of a handwritten digit and the model will predict the number.')

    # File uploader
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction)

        st.write(f'Prediction: {predicted_class}')

if __name__ == '__main__':
    main()
