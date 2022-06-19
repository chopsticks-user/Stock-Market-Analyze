import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def change(data = None, interval = 1, percentage = False):
    result = np.zeros(len(data))
    for i, price in enumerate(data[1:], start = 1):
        result[i] = price - data[i - 1]
        result[i] /= data[i - 1] if percentage else 1.0
    return result

