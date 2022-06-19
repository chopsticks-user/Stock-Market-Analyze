import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk
import os
import time

import sklearn
from indicators import Indicators
from tools import change
from sklearn import linear_model
from sklearn.utils import shuffle
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
if __name__ == "__main__":
    stocks_path = "Stock/Data/Stocks"
    etfs_path = "Stock/Data/ETFs"
    stocks_list = os.listdir(stocks_path)
    etfs_list = os.listdir(etfs_path)

    data_path = stocks_path + "/" + stocks_list[0]
    data = pd.read_csv(data_path)
    data = data.drop(columns = ["OpenInt"])
    data = data.dropna()
    transposed_data = data.T

    # print(transposed_data[0]["Close"])
    # print(data.head())
    # print(data.shape)
    # print(data.columns)
    # print(data.dtypes)
    # print(data.describe(include = "all", percentiles = [.25, .5, .75, .9, .99]))

    indicators = Indicators(len(data), [20, 50, 200])
    for i in range(len(data)):
        indicators.update(*transposed_data[i])
        #print(indicators.data["MACD"][-1])

    indicators_list = indicators.list

    processed_data = pd.DataFrame({
        "Profit": change(data["Close"]), 
        "SMA 20": change(np.array((indicators.data["SMA 20"]))), 
        "SMA 50": change(np.array((indicators.data["SMA 50"]))), 
        "SMA 200": change(np.array((indicators.data["SMA 200"]))), 
        "WMA 20": change(np.array((indicators.data["WMA 20"]))), 
        "WMA 50": change(np.array((indicators.data["WMA 50"]))), 
        "WMA 200": change(np.array((indicators.data["WMA 200"]))), 
        "EMA 20": change(np.array((indicators.data["EMA 20"]))), 
        "EMA 50": change(np.array((indicators.data["EMA 50"]))), 
        "EMA 200": change(np.array((indicators.data["EMA 200"]))), 
        "Aroon Up": change(np.array((indicators.data["Aroon Up"]))),  
        "Aroon Down": change(np.array((indicators.data["Aroon Down"]))), 
        "MACD": change(np.array((indicators.data["MACD"]))), 
        "Momentum": change(np.array((indicators.data["Momentum"]))),  
        "Stoch K": change(np.array((indicators.data["Stoch K"]))),  
        "Stoch D": change(np.array((indicators.data["Stoch D"]))),  
        "William R": change(np.array((indicators.data["William R"]))),  
        "AD": change(np.array((indicators.data["AD"])))
    })
    #print(processed_data)

    corr = processed_data.corr()
    print(corr["Profit"])

    drop_list = ["Profit"]
    # for column in corr.columns[1:]:
    #     if corr["Profit"][column] < 0.2:
    #         drop_list.append(column)
    
    X = np.array(processed_data.drop(columns = drop_list))
    y = np.array(processed_data[["Profit"]])

    lin_reg = linear_model.LinearRegression()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.5)
    lin_reg.fit(X_train, y_train)
    accuracy = lin_reg.score(X_test, y_test)

    y_pred = lin_reg.predict(X_test)
    errors = np.zeros(len(y_pred))
    for i in range(len(y_pred)):
        print("Predicted: {}, Value: {}, Error: {}%".format(y_pred[i], y_test[i], 
        200 * abs((y_pred[i] - y_test[i]) / (y_pred[i] + y_test[i]))))
        errors[i] = 200 * abs((y_pred[i] - y_test[i]) / (y_pred[i] + y_test[i]))

    print(accuracy)
    errors = pd.DataFrame(errors)
    print(errors.describe(include = "all", percentiles = [.25, .5, .75, .9, .99]))




    # sns.displot(errors)
    # plt.axis([0.0, 1000, 0.0, 500])
    # plt.show(block = True)



    