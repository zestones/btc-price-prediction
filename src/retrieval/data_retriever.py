import requests
import re
import os

class DataRetriever:
    __available_datasets = ['blocks', 'transactions', 'inputs', 'outputs', 'addresses']
    __special_datasets = ['tokens', 'calls']
    __available_coins = ['bitcoin', 'bitcoin-cash', 'dogecoin', 'ethereum', 'dash', 'zcash', 'litecoin']
    
    def __init__(self, coin, dataset_name):
        self.coin = coin
        self.dataset_name = dataset_name
        self.base_url = "https://gz.blockchair.com/{coin}/{dataset_name}"
        
        # check if the coin is available
        if self.coin not in self.__available_coins:
            raise ValueError('Coin not available')

        # check if the dataset is available
        if self.dataset_name not in self.__available_datasets:
            raise ValueError('Dataset not available')
        
        # check if the dataset is special
        if self.coin != 'ethereum' and self.dataset_name in self.__special_datasets:
            raise ValueError('Dataset not available for this coin') 
        
        self.url = self.base_url.format(coin=self.coin, dataset_name=self.dataset_name)

    def _scrape_url(self):
        response = requests.get(self.url)
        if response.status_code != 200:
            raise ValueError('Error while retrieving the dataset urls')
                        
        files_names = re.findall(r'href=[\'"]?([^\'" >]+)', response.text)
        files_names = [file for file in files_names if self.dataset_name in file]

        return files_names
    
    def save_dataset(self, base_path):
        base_path = f'{base_path}/{self.coin}/{self.dataset_name}/'
        os.makedirs(base_path, exist_ok=True)
        
        files_names = self._scrape_url()
        for file_name in files_names:
            response = requests.get(self.url + "/" + file_name, stream=True)
            
            if response.status_code != 200: 
                raise ValueError('Error while retrieving the dataset')
            
            with open(base_path + "/" + file_name, 'wb') as f:
                f.write(response.content)