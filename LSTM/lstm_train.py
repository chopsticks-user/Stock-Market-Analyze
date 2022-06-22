import numpy as np
import pandas as pd
import torch as T
import os
import time

from LSTM.lstm_indicators import LSTMIndicators
from LSTM.lstm_model import LSTMModel
from tools import *

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def lstm_train():
    stocks_path = "Stock/Data/Stocks"
    etfs_path = "Stock/Data/ETFs"
    stocks_list = os.listdir(stocks_path)
    etfs_list = os.listdir(etfs_path)

    # print(inputs)

    # corr = processed_data.corr()
    # print(corr["Profit"])

    # print(processed_data[["Profit"]])

    model_path = "Load/Model/lstm_model_0619"
    load = True

    for stock_index in range(217, len(stocks_list)):
        data_path = stocks_path + "/" + stocks_list[stock_index]
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

        indicators = LSTMIndicators(len(data), [20, 50, 200])
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

        if load:
            lstm = train_load(model_path)
        else:
            lstm = LSTMModel(in_features_dim = len(indicators_list), hidden_dim = 32, out_features_dim = 1, file_path = model_path)

        targets = T.tensor(np.array(processed_data[["Profit"]]), dtype = T.float).to(lstm.device)
        inputs = T.tensor(np.array(processed_data.drop(columns = ["Profit"])), dtype = T.float).to(lstm.device)

        # time.sleep(10)
        average_loss = 0.0
        average_mape = 0.0
        total_step = 0
        capital = 1000.0
        position = 0.0
        sequence_length = 10
        future = 1
        for index in range(sequence_length, len(processed_data) - future):
            prediction = lstm.forward(inputs[index - sequence_length : index])
            target = targets[index + future]
            # if abs(target) < 0.005:
            #     total_step -= 1
            #     mape = 0.0
            # else:
            #     mape = abs((target - prediction) / target)

            if target == 0.0:
                total_step -= 1
                mape = 0.0
            else:
                mape = abs((target - prediction) / target)

            loss = lstm.backward(target)
            average_loss += loss
            average_mape += mape
            total_step += 1
            if total_step % 100 == 0:
                print(f"    Targets = {targets[index + future]}, Predicted = {prediction}, mape = {mape}")

        average_loss /= total_step
        average_mape /= total_step
        print(f"Epoch: {stock_index}, Average Loss: {average_loss}, Average MAPE: {average_mape}")
        train_save(lstm)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''



    