import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
from tools import *

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
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

