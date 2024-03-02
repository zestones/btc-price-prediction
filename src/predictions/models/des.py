import numpy as np
import pandas as pd
from predictions.models.model import Model

class DES(Model):
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [
            np.random.randn(input_size, layer_size),
            np.random.randn(layer_size, output_size),
            np.random.randn(layer_size, 1),
            np.random.randn(1, layer_size),
        ]
                
    def fit(self, **kwargs):
        raise NotImplementedError("This model is meant to be trained using the DESAgent class.")
    
    def evaluate(self, **kwargs):
        raise NotImplementedError("This model cannot be evaluated using the evaluate method. Use the DESAgent class to evaluate it.")
       
    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        decision = np.dot(feed, self.weights[1])
        buy = np.dot(feed, self.weights[2])
        return decision, buy

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights
        
    def save(self, path):
        np.savez(path, *self.weights)
    
    @staticmethod
    def load(path):
        weights = np.load(path)
        model = DES(0, 0, 0)
        model.set_weights([weights[key] for key in weights])
        return model
