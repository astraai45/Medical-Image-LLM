import numpy as np
import tensorflow as tf
from utils import load_new_data, create_keras_model, preprocess_image, save_model_weights, load_and_preprocess_data
from client import MedicalMNISTClient
import flwr as fl
import os
from sklearn.model_selection import train_test_split


# Set random seed
tf.random.set_seed(42)
np.random.seed(42)

# Hardcoded label mapping
label_mapping = {'AbdomenCT': 0, 'BreastMRI': 1, 'Hand': 2, 'CXR': 3, 'HeadCT': 4, 'ChestCT': 5}

# Load new data
new_data_path = './new_medical_data'
if os.path.exists(new_data_path):
    print("Loading new data from directory.")
    new_x, new_y = load_new_data(new_data_path, label_mapping=label_mapping)
    print("Custom new data loaded.")
else:
    # Check for saved test subset
    if os.path.exists('test_images.npy') and os.path.exists('test_labels.npy'):
        print("Loading saved test subset.")
        new_x = np.load('test_images.npy')
        new_y = np.load('test_labels.npy')
    else:
        print("New data directory not found. Creating test subset from medical_mnist.")
        images, labels, _ = load_and_preprocess_data('./medical_mnist', limit_samples=200)
        _, new_x, _, new_y = train_test_split(images, labels, test_size=0.5, random_state=42)
        np.save('test_images.npy', new_x)
        np.save('test_labels.npy', new_y)
print(f"New data shape: {new_x.shape}, {new_y.shape}")

# Create and start client
client = MedicalMNISTClient(new_x, new_y)
print("Starting Flower client...")
fl.client.start_client(server_address="localhost:8080", client=client.to_client())

# Get updated weights from client
updated_weights = client.get_parameters(config={})

# Update and save global model
print("\nUpdating global model with client's new weights:")
global_model = create_keras_model()
global_model.set_weights(updated_weights)
global_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
save_model_weights(global_model, file_path='model.weights.h5')

# Evaluate global model on new data
print("\nEvaluating global model on new data...")
test_dataset = tf.data.Dataset.from_tensor_slices((new_x[..., np.newaxis], new_y)).batch(32)
loss, accuracy = global_model.evaluate(test_dataset, verbose=0)
print(f'Global Model - Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Predict on external image
external_image_path = 'images/head5.jpg'
external_image = preprocess_image(external_image_path)
if external_image is not None:
    external_image = np.expand_dims(external_image, axis=0)
    predictions = global_model.predict(external_image)
    predicted_label = np.argmax(predictions, axis=1)[0]
    class_mapping = {0: 'AbdomenCT', 1: 'BreastMRI', 2: 'Hand', 3: 'CXR', 4: 'HeadCT', 5: 'ChestCT'}
    print(f'Predicted Class for External Image: {class_mapping[predicted_label]}')
else:
    print("Error: Could not load external image.")