import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# Load the saved model
MODEL_PATH = 'best_model.h5'
model = load_model(MODEL_PATH)

# Class mapping
class_mapping = {
    0: 'AbdomenCT',
    1: 'BreastMRI',
    2: 'Hand',
    3: 'CXR',
    4: 'HeadCT',
    5: 'ChestCT'
}

# Privacy info mapping
privacy_info = {
    'HeadCT': {"Privacy Level": "High", "Privacy Budget (ε)": "0.5", "Noise Level": "High noise"},
    'BreastMRI': {"Privacy Level": "High", "Privacy Budget (ε)": "0.5", "Noise Level": "High noise"},
    'AbdomenCT': {"Privacy Level": "Medium", "Privacy Budget (ε)": "1.0", "Noise Level": "Moderate noise"},
    'ChestCT': {"Privacy Level": "Medium", "Privacy Budget (ε)": "1.0", "Noise Level": "Moderate noise"},
    'CXR': {"Privacy Level": "Low", "Privacy Budget (ε)": "2.0", "Noise Level": "Minimal noise"},
    'Hand': {"Privacy Level": "Low", "Privacy Budget (ε)": "2.0", "Noise Level": "Minimal noise"}
}

# Streamlit app title
st.title("Medical Image Classification with Privacy Info")
st.write("Upload a medical image to classify it and view its privacy details.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((64, 64))
    image = np.array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)   # Add batch dimension

    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    predictions = model.predict(image)
    predicted_label = np.argmax(predictions, axis=1)[0]
    predicted_class = class_mapping[predicted_label]

    # Display prediction
    st.write("## Prediction:")
    st.write(f"**Predicted class:** {predicted_class}")

    # Display corresponding privacy info
    if predicted_class in privacy_info:
        st.write("## Privacy Information:")
        info = privacy_info[predicted_class]
        st.write(f"**Privacy Level:** {info['Privacy Level']}")
        st.write(f"**Privacy Budget (ε):** {info['Privacy Budget (ε)']}")
        st.write(f"**Noise Level:** {info['Noise Level']}")
    else:
        st.warning("Privacy information not available for this class.")

# Run the app using:
# streamlit run app.py
