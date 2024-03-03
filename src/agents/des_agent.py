from agents.strategies.deep_evolution_strategy import Deep_Evolution_Strategy as DES
import matplotlib.pyplot as plt
from typing import Tuple, List
import numpy as np
import tensorflow as tf


class DataHandler:
    def __init__(self, data_points: List[float], window_size: int, skip) -> None:
        self.data_points = tf.constant(data_points, dtype=tf.float32)
        self.train = data_points
        self.test = data_points
        
        self.window_size = window_size
        
        self.length_train = len(self.train) - 1
        self.length_test = len(self.test) - 1
        
        self.skip = skip
        
    def train_test_split(self, test_size: float = 0.2) -> None:
        """
        Splits the data into training and testing sets.

        Args:
            test_size (float): The size of the testing set.
        """
        train_size = int(len(self.data_points) * (1 - test_size))
        
        train_data = self.data_points[:train_size]
        test_data = self.data_points[train_size:]
        
        self.set_train_data(train_data)
        self.set_test_data(test_data)
              
    def get_state(self, data: tf.Tensor, t: int, n: int) -> tf.Tensor:
        """
        Get the state for the given data, time step, and sequence length.

        Parameters:
        - data: The input data tensor.
        - t: The current time step.
        - n: The sequence length.

        Returns:
        - state: The calculated state as a TensorFlow tensor.
        """
        d = t - n + 1
        block = tf.gather(data, indices=tf.range(d, t + 1), axis=0) if d >= 0 else tf.concat([-d * [data[0]], data[:t + 1]], axis=0)
        res = tf.subtract(block[1:], block[:-1])
        return tf.expand_dims(res, axis=0)
    
    def set_train_data(self, data: List[float]) -> None:
        """
        Sets the training data.
        """
        self.train = tf.constant(data, dtype=tf.float32)
        self.length_train = len(data) - 1
        
    def set_test_data(self, data: List[float]) -> None:
        """
        Sets the testing data.
        """
        self.test = tf.constant(data, dtype=tf.float32)
        self.length_test = len(data) - 1

    def set_window_size(self, window_size: int) -> None:
        """
        Sets the window size.
        """
        self.window_size = window_size
        
    def set_skip(self, skip: int) -> None:
        """
        Sets the skip value.
        """
        self.skip = skip

class DESAgent(DataHandler):

    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03
    BUY_ACTION = 1
    SELL_ACTION = 2
    
    window_size = 30
    skip = 1

    def __init__(self, 
                 model: DES,
                 money: int,
                 max_buy: int,
                 max_sell: int,
                 data_points: List[float],
                 window_size: int,
                 skip: int = 1
                 ) -> None:
        super().__init__(data_points, window_size, skip)
        self.model = model
        
        self.initial_money = money
        self.max_buy = max_buy
        self.max_sell = max_sell
        
        self.es = DES(
            self.model.get_weights(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )
                
    def fit(self, iterations: int, checkpoint: int) -> None:
            """
            Trains the agent using the Evolution Strategy algorithm.

            Args:
                iterations (int): The number of iterations to train the agent.
                checkpoint (int): The interval at which to print training progress.

            Returns:
                None
            """
            self.es.train(iterations, print_every=checkpoint)

    def act(self, sequence: tf.Tensor) -> Tuple[int, float]:
        decision, buy = self.model.predict(sequence)
        return tf.argmax(decision[0]), float(buy[0])
    
    def _calculate_buy_units(self, initial_money: float, price: tf.Tensor, buy: float, t: int) -> float:
        if buy < 0:
            # buy unit is 10% of what you can afford
            buy_units = (initial_money * 0.1) / price[t]
        elif (buy * price[t]) > initial_money or buy > self.max_buy:
            # if we want to buy more than we can afford
            # restrict buy_units to the maximum we can buy with the money we have
            # without going into debt and without exceeding the maximum buy limit
            buy_units = tf.minimum((initial_money * 0.9) / price[t], self.max_buy)
        else:
            buy_units = buy
            
        return buy_units
    
    def _calculate_sell_units(self, quantity: float) -> float:
        return tf.minimum(self.max_sell, quantity)
    
    def get_reward(self, weights: tf.Tensor) -> float:
        initial_money = self.initial_money
        starting_money = initial_money
        
        self.model.weights = weights
        state = self.get_state(self.train, 0, self.window_size + 1)
        
        inventory = tf.constant([], dtype=tf.float32)
        quantity = tf.constant(0, dtype=tf.float32)
        
        for t in range(0, self.length_train, self.skip):
            action, buy = self.act(state)
            next_state = self.get_state(self.train, t + 1, self.window_size + 1)
            
            if action == self.BUY_ACTION and initial_money > 0:
                buy_units = self._calculate_buy_units(initial_money, self.train, buy, t)
                
                total_buy = buy_units * self.train[t]
                initial_money -= total_buy
                inventory = tf.concat([inventory, [total_buy]], axis=0)
                quantity += buy_units
                
            elif action == self.SELL_ACTION and tf.size(inventory) > 0:
                sell_units = self._calculate_sell_units(quantity)
                    
                quantity -= sell_units
                total_sell = sell_units * self.train[t]
                initial_money += total_sell

            state = next_state
            
        return tf.divide((initial_money - starting_money), starting_money) * 100

    def buy(self):
        initial_money = self.initial_money
        state = self.get_state(self.test, 0, self.window_size + 1)
        starting_money = initial_money
        
        states_sell = []
        states_buy = []

        inventory = []
        quantity = 0
        
        for t in range(0, self.length_test, self.skip):
            action, buy = self.act(state)
            next_state = self.get_state(self.test, t + 1, self.window_size + 1)
            if action == self.BUY_ACTION and initial_money > 0:
                buy_units = self._calculate_buy_units(initial_money, self.test, buy, t)
                    
                total_buy = buy_units * self.test[t]
                initial_money -= total_buy
                inventory.append(total_buy)
                quantity += buy_units
                states_buy.append(t)
                
                print(
                    'day %d: buy %f units at price %f, total balance %f'
                    % (t, buy_units, total_buy, initial_money)
                )
                
            elif action == self.SELL_ACTION and len(inventory) > 0:
                bought_price = inventory.pop(0)
                sell_units = self._calculate_sell_units(quantity)
                
                if sell_units <= 0:
                    continue
                   
                quantity -= sell_units
                total_sell = sell_units * self.test[t]
                initial_money += total_sell
                states_sell.append(t)
                
                try:
                    invest = ((total_sell - bought_price) / bought_price) * 100
                except:
                    invest = 0
                
                print(
                    'day %d, sell %f units at price %f, investment %f %%, total balance %f,'
                    % (t, sell_units, total_sell, invest, initial_money)
                )
                
            state = next_state

        invest = ((initial_money - starting_money) / starting_money) * 100
        print(
            '\ntotal gained %f, total investment %f %%'
            % (initial_money - starting_money, invest)
        )
        plt.figure(figsize = (20, 10))
        plt.plot(self.test, label = 'true price', c = 'g')
        plt.plot(self.test, 'X', label = 'predict buy', markevery = states_buy, c = 'b')
        plt.plot(self.test, 'o', label = 'predict sell', markevery = states_sell, c = 'r')
        plt.legend()
        plt.show()