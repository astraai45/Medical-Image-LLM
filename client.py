import flwr as fl
import tensorflow as tf
import numpy as np
from utils import create_keras_model

class MedicalMNISTClient(fl.client.NumPyClient):
    def __init__(self, new_x, new_y):
        self.model = create_keras_model()
        self.new_x = new_x
        self.new_y = new_y
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.new_x[..., np.newaxis], self.new_y, epochs=1, batch_size=32, verbose=0)
        print("Client trained on new data.")
        return self.model.get_weights(), len(self.new_x), {}
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.new_x[..., np.newaxis], self.new_y, verbose=0)
        return loss, len(self.new_x), {"accuracy": accuracy, "num_examples": len(self.new_x)}