from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class ETL:
    """
    Extract, transform and load data from a csv file.
    Creates a train and test set, 
    and splits the data into windows of a given timestep.
    """
    def __init__(self, path: str, features: list, test_size: float = 0.2, timestep: int = 6):
        self.path = path
        self.features = features
        self.test_size = test_size
        self.timestep = timestep
        
        self._scaler = MinMaxScaler(feature_range=(0, 1))

        self.train, self.test = self.extract_transform_load()
        self.train_x, self.train_y = self._window(self.train)
        self.test_x, self.test_y = self._window(self.test) 
        
    def extract_transform_load(self) -> (np.array, np.array):
        df = self._load()
        data_values = self._extract(df)
        train, test = self._transform(data_values)
        return train, test

    def _extract(self, df: pd.DataFrame) -> np.array:
        return df[self.features].values
    
    def _transform(self, data: pd.DataFrame) -> (np.array, np.array):
        data_scaled = self.scale(data)
        return self._train_test_split(data_scaled)
    
    def _train_test_split(self, data: np.array) -> (np.array, np.array):
        train_size = int(len(data) * (1 - self.test_size))
        return data[:train_size], data[train_size:]
    
    def _window(self, data: pd.DataFrame) -> np.array:
        x, y = [], []
        total_length = len(data) // (self.timestep + 1)
        
        for i in range(total_length):
            start = i * (self.timestep + 1)
            end = start + self.timestep
            
            x.append(data[start:end, :])
            y.append(data[end, 0])
            
        return np.array(x), np.array(y).reshape(-1, 1)

    def scale(self, df):
        return self._scaler.fit_transform(df)
    
    def _load(self):
        return pd.read_csv(self.path)
    
    def _reshape_data(self, data):
        adjusted = np.zeros((data.shape[0], len(self.features)))
        adjusted[:, 0] = data.ravel()
        return adjusted
    
    def inverse_scale(self, data):
        reshaped_data = self._reshape_data(data)
        inversed_data = self._scaler.inverse_transform(reshaped_data)
        return inversed_data[:, 0]