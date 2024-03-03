import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple, List
from predictions.models.model import Model

class DES(Model):
    def __init__(self, input_size: int, layer_size: int, output_size: int):
        self.weights = [
            tf.random.normal((input_size, layer_size)),
            tf.random.normal((layer_size, output_size)),
            tf.random.normal((layer_size, 1)),
            tf.random.normal((1, layer_size)),
        ]
                
    def fit(self, **kwargs):
        raise NotImplementedError("This model is meant to be trained using the DESAgent class.")
    
    def evaluate(self, **kwargs):
        raise NotImplementedError("This model cannot be evaluated using the evaluate method. Use the DESAgent class to evaluate it.")
       
    def predict(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        feed = tf.matmul(inputs, self.weights[0]) + self.weights[-1]
        decision = tf.matmul(feed, self.weights[1])
        buy = tf.matmul(feed, self.weights[2])
        return decision, buy

    def get_weights(self) -> List[tf.Tensor]:
        return self.weights

    def set_weights(self, weights: List[tf.Tensor]) -> None:
        self.weights = weights
        
    def save(self, path):
        np.savez(path, *self.weights)
    
    @staticmethod
    def load(path):
        weights = np.load(path)
        model = DES(0, 0, 0)
        model.set_weights([weights[key] for key in weights])
        return model
