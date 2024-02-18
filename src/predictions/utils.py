import tensorflow as tf
import numpy as np
import random
import os

def deterministic_mode():
    SEED = 42

    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'