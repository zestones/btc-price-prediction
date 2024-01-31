class Predict:
    """
    Class to predict the output of the model
    """
    def __init__(self, model, test_x, test_y) -> None:
        self.model = model
        self.test_x = test_x
        self.test_y = test_y
        self.predictions = self.predict()
    
    def predict(self):
        return self.model.predict(self.test_x)