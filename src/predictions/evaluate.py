from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from matplotlib import pyplot as plt
import numpy as np
import tabulate

class Evaluate:
  """
  Evaluate the model with different metrics, 
  given the actual and predicted values.
  """
  def __init__(self, actual, predictions) -> None:
    self.actual = actual
    self.predictions = predictions
    self.var_ratio = self.compare_var()
    self.mape = self.evaluate_model_with_mape()
    self.mse = self.evaluate_model_with_mse()
    self.rmse = self.evaluate_model_with_rmse()
    self.mae = self.evaluate_model_with_mae()
    self.r2 = self.evaluate_model_with_r2()

  def evaluate_model_with_r2(self):
    return r2_score(self.actual, self.predictions)
  
  def evaluate_model_with_mae(self):
    return mean_absolute_error(self.actual, self.predictions)
  
  def evaluate_model_with_mse(self):
    return mean_squared_error(self.actual, self.predictions)
  
  def evaluate_model_with_rmse(self):
    return mean_squared_error(self.actual, self.predictions, squared=False)
  
  def compare_var(self):
    return abs(1 - (np.var(self.predictions) / np.var(self.actual)))

  def evaluate_model_with_mape(self):
    return mean_absolute_percentage_error(self.actual.flatten(), self.predictions.flatten())
  
  def plot(self):
    plt.figure(figsize=(12, 6))

    time_range = range(len(self.actual))

    # Plotting the actual prices
    plt.plot(time_range, self.actual, label='Real Price', color='blue', marker='o', linestyle='-')

    # Plotting the predicted prices
    plt.plot(time_range, self.predictions, label='Predicted Price', color='orange', marker='o', linestyle='--')

    # Highlighting the difference between actual and predicted prices
    diff = np.abs(np.array(self.actual) - np.array(self.predictions))
    plt.bar(time_range, diff, alpha=0.3, color='red', label='Absolute Price Difference')

    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()
  
  def print(self):
    table = [
      ['MSE', self.mse],
      ['RMSE', self.rmse],
      ['MAE', self.mae],
      ['R2', self.r2],
      ['MAPE', self.mape],
      ['Variance Ratio', self.var_ratio]
    ]
    print(tabulate.tabulate(table, headers=['Metric', 'Value'], tablefmt='github'))
