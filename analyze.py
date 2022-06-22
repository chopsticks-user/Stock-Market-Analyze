import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

def change(data):
    changes = np.zeros(len(data))
    for i in range(1, len(data)):
        if data[i] != data[i - 1]:
            changes[i] = (data[i] - data[i - 1]) / data[i - 1]
    return changes

stocks_path = "Stock/Data/Stocks"
etfs_path = "Stock/Data/ETFs"
stocks_list = os.listdir(stocks_path)
etfs_list = os.listdir(etfs_path)

data = pd.read_csv(stocks_path + "/" + stocks_list[500])
data = data.dropna()

price_changes = change(np.asarray(data[["Close"]]))
print(price_changes)

sns.distplot(price_changes)
plt.show(block = True)

