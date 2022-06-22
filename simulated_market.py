import numpy as np
import math
from tools import *

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class SimulatedMarket(object):
    def __init__(self, data = None, initial_price = 100.0):
        self.data = data
        self.data_size = len(data)
        self.initial_price = initial_price if initial_price != 0.0 else 100.0
        self.current_price = self.initial_price
        self.current_price_movement = 0.0001

        self.current_step = 0

    def update(self):
        self.current_price_movement = self.data[self.current_step % self.data_size]
        self.current_price *= (1 + self.current_price_movement)
        self.current_step += 1
        return self.current_price, self.current_price_movement

