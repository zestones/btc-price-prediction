import joblib
import tensorflow as tf

class Model:
    def __init__(self, model):
        self.model = model
        
    def compile(self, loss='mean_squared_error', optimizer='adam'):
        self.model.compile(loss=loss, optimizer=optimizer)

    def fit(self, X, y, epochs=10, batch_size=32):
        self.model.fit(X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            shuffle=False,
            verbose=2,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)]
        )

    def predict(self, data):
        return self.model.predict(data)
    
    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def save(self, path):
        joblib.dump(self.model, path)

    @staticmethod
    def load(path):
        return Model(joblib.load(path))