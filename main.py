import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk
import os
import time
from indicators import Indicators

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
if __name__ == "__main__":
    stocks_path = "Stock/Data/Stocks"
    etfs_path = "Stock/Data/ETFs"
    stocks_list = os.listdir(stocks_path)
    etfs_list = os.listdir(etfs_path)

    data_path = stocks_path + "/" + stocks_list[0]
    data = pd.read_csv(data_path)
    data = data.drop(columns = ["OpenInt"])
    transposed_data = data.T
    print(transposed_data[0]["Close"])

    print(data.head())
    print(data.shape)
    print(data.columns)
    print(data.dtypes)
    print(data.describe(include = "all", percentiles = [.25, .5, .75, .9, .99]))
    # sns.displot(data[["Close"]][4421:])
    # plt.show(block = True)
    # for c in data.columns[1:]:
    #     sns.displot(data[c][4000:])
    #     plt.show(block = True)

    indicators = Indicators(len(data), [20, 50, 200])
    for i in range(len(data)):
        print(indicators.update(*transposed_data[i]))
        #time.sleep(2)
    print(indicators.memory_count)

    

    