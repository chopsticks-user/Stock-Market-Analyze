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

    data_path = stocks_path + "/" + np.random.choice(stocks_list)
    data = pd.read_csv(data_path)
    data = data.drop(columns = ["OpenInt"])
    transposed_data = data.T
    print(transposed_data[0]["Close"])

    print(data.head())
    print(data.shape)
    print(data.columns)
    print(data.dtypes)
    print(data.describe(include = "all", percentiles = [.25, .5, .75, .9, .99]))

    indicators = Indicators(len(data), [20, 50, 200])
    for i in range(len(data)):
        indicators.update(*transposed_data[i])
        #print(indicators.data["MACD"][-1])

    indicators_list = indicators.list

    processed_data = pd.DataFrame({
        "Close": data["Close"], 
        "SMA 20": list(indicators.data["SMA 20"]), 
        "SMA 50": list(indicators.data["SMA 50"]), 
        "SMA 200": list(indicators.data["SMA 200"]), 
        "WMA 20": list(indicators.data["WMA 20"]), 
        "WMA 50": list(indicators.data["WMA 50"]), 
        "WMA 200": list(indicators.data["WMA 200"]),
        "EMA 20": list(indicators.data["EMA 20"]), 
        "EMA 50": list(indicators.data["EMA 50"]), 
        "EMA 200": list(indicators.data["EMA 200"]), 
    })
    #print(processed_data)

    corr = processed_data.corr()
    print(corr["Close"])

    # sns.displot(data[["Close"]][4421:])
    # plt.show(block = True)
    # for c in data.columns[1:]:
    #     sns.displot(data[c][4000:])
    #     plt.show(block = True)

    

    