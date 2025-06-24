# Medical-Image-LLM.


Here's a structured GitHub README for your medical image analysis project that you can use when uploading to GitHub:

---

# Medical Image Analysis with Privacy and Chatbot Integration

This project focuses on **medical image classification** combined with **privacy information** and a **medical chatbot**. It allows users to upload medical images and get predictions, privacy details, and consult with a chatbot to ask medical-related questions. The project uses various technologies like **Streamlit**, **Keras**, **TensorFlow**, **Langchain**, and **Flower**.

## Features

* **Image Classification**: Upload a medical image and get predictions on the class (e.g., AbdomenCT, BreastMRI, etc.) and corresponding privacy information.
* **Medical Chatbot**: Ask a medical question and retrieve answers based on stored knowledge from PDFs.
* **Federated Learning Integration**: A client model is designed to participate in federated learning using **Flower** to train a global model while maintaining data privacy.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Model Details](#model-details)
5. [Federated Learning](#federated-learning)
6. [Contributing](#contributing)
7. [License](#license)

## Requirements

* Python 3.x
* TensorFlow
* Keras
* OpenCV
* PIL (Pillow)
* Streamlit
* Langchain
* Flower
* Chroma
* PDFPlumber
* Numpy
* Scikit-learn

## Installation

To install the dependencies, use the following command:

```bash
pip install -r requirements.txt
```

You may need to install other dependencies depending on your environment, such as `opencv-python-headless` for headless environments.

## Usage

1. **Image Classification**:

   * Run the Streamlit app using the following command:

     ```bash
     streamlit run app.py
     ```

   * Upload a medical image (e.g., a CT scan, MRI) in supported formats (`.jpg`, `.jpeg`, `.png`).

   * The model will predict the image class and display privacy details about that class.

2. **Medical Chatbot**:

   * Navigate to the **Medical Chatbot** tab in the app.
   * Enter a medical question, and the chatbot will try to provide an answer based on the preloaded knowledge PDFs.

## Model Details

* The model used for **image classification** is a custom deep learning model built using Keras and TensorFlow.
* **Class Mapping**:

  * 0: AbdomenCT
  * 1: BreastMRI
  * 2: Hand
  * 3: CXR
  * 4: HeadCT
  * 5: ChestCT
* The privacy information is mapped according to the class predictions, with details like **Privacy Level**, **Privacy Budget**, and **Noise Level**.

## Federated Learning

This project integrates **Federated Learning** using **Flower (FL)**. The local model is trained on new data and sends its updated weights to the server. The server combines the updates to improve the global model while keeping the local data secure.

### Running the Federated Client:

To start the Flower client and train on new data:

1. Make sure you have a running Flower server (usually at `localhost:8080`).
2. Run the federated learning client with:

   ```bash
   python federated_client.py
   ```

This will update the global model using new data and evaluate its performance.

## Contributing

We welcome contributions to improve this project! Please feel free to open issues, submit pull requests, or provide feedback.

### Contributors

* [Balaji-Kartheek](https://github.com/Balaji-Kartheek)
* [Pavanbalaji45](https://github.com/pavanbalaji45)

