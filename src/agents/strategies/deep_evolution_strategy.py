import time
import numpy as np
import tensorflow as tf
from typing import List, Callable

class Deep_Evolution_Strategy:
    """
    A class representing the Deep Evolution Strategy algorithm.

    Attributes:
        weights (List[np.ndarray]): The initial weights for the algorithm.
        reward_function (Callable[[List[np.ndarray]], float]): The reward function used to evaluate the weights.
        population_size (int): The size of the population.
        sigma (float): The standard deviation for the mutation.
        learning_rate (float): The learning rate for updating the weights.

    Methods:
        _get_weight_from_population(weights, population): Returns the weights after applying mutation.
        get_weights(): Returns the current weights.
        train(epoch, print_every): Trains the algorithm for a specified number of epochs.
    """

    inputs = None

    def __init__(
        self,
        weights: List[tf.Tensor],
        reward_function: Callable[[List[tf.Tensor]], float],
        population_size: int,
        sigma: float,
        learning_rate: float
    ) -> None:
        """
        Initializes the Deep_Evolution_Strategy class.

        Args:
            weights (List[np.ndarray]): The initial weights for the algorithm.
            reward_function (Callable[[List[np.ndarray]], float]): The reward function used to evaluate the weights.
            population_size (int): The size of the population.
            sigma (float): The standard deviation for the mutation.
            learning_rate (float): The learning rate for updating the weights.

        Returns:
            None
        """
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights: List[tf.Tensor], population: List[tf.Tensor]) -> List[tf.Tensor]:
        """
        Returns the weights after applying mutation.

        Args:
            weights (List[tf.Tensor]): The current weights.
            population (List[List[tf.Tensor]]): The population of mutated weights.

        Returns:
            List[tf.Tensor]: The weights after applying mutation.
        """
        weights_population = []
        for index, i in enumerate(population):
            jittered = tf.multiply(self.sigma, i)
            weights_population.append(weights[index] + jittered)
            
        return weights_population

    def get_weights(self) -> List[tf.Tensor]:
        """
        Returns the current weights.

        Returns:
            List[tf.Tensor]: The current weights.
        """
        return self.weights
    
    def _generate_individual(self) -> List[tf.Tensor]:
        """
        Generates an individual for the deep evolution strategy.

        Returns:
            List[tf.Tensor]: The generated individual.
        """
        x = []
        for w in self.weights:
            x.append(tf.random.normal(shape=w.shape))
        return x
    
    def train(self, epoch: int = 100, print_every: int = 1) -> None:
        """
        Trains the algorithm for a specified number of epochs.

        Args:
            epoch (int): The number of epochs to train the algorithm. Default is 100.
            print_every (int): The frequency of printing the reward during training. Default is 1.

        Returns:
            None
        """
        lasttime = time.time()
        for i in range(epoch):
            rewards = tf.zeros(self.population_size)
            population = []

            for k in range(self.population_size):
                population.append(self._generate_individual())
                weights_population = self._get_weight_from_population(self.weights, population[k])
                rewards = tf.tensor_scatter_nd_add(rewards, [[k]], [self.reward_function(weights_population)])


            updated_weights = []
            rewards = (rewards - tf.reduce_mean(rewards)) / tf.math.reduce_std(rewards)
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                weight_update = (
                    w 
                    + self.learning_rate 
                    / (self.population_size * self.sigma)
                    *  np.dot(A.T, rewards).T
                )
                updated_weights.append(weight_update)
                
            self.weights = updated_weights

            if (i + 1) % print_every == 0:
                print('iter %d. reward: %f' % (i + 1, self.reward_function(self.weights)))

        print('time taken to train:', time.time() - lasttime, 'seconds')
