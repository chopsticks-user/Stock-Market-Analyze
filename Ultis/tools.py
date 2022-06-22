import numpy as np
import pandas as pd
import math
import torch as T
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# tools for calculating

def change(data = None, interval = 1, percentage = True):
    result = np.zeros(len(data))
    for i, price in enumerate(data[1:], start = 1):
        result[i] = price - data[i - 1]
        if data[i - 1] == 0:
            result[i] = result[i - 1]
            continue
        result[i] /= data[i - 1] if percentage else 1.0
    return result

def change(data):
    changes = np.zeros(len(data))
    for i in range(1, len(data)):
        if data[i] != data[i - 1] and data[i - 1] != 0.0:
            changes[i] = (data[i] - data[i - 1]) / data[i - 1]
    return changes

def random_price_movement():
    return np.tanh(np.random.rand() - 0.46) / 10

def position(pre_position, price_movement):
    return pre_position * (1 + price_movement)

def trailing_stop_loss(current_position, last_stop_loss, inital_stop_loss, loss_rate):
    return max(current_position * (1 - loss_rate), inital_stop_loss, last_stop_loss)

def trigger_stop_loss(stop_loss, current_position, acceptable_margin):
    return current_position <= stop_loss or abs(current_position - stop_loss) <= acceptable_margin * stop_loss

def adjust_loss_rate(loss_rate, min_loss_rate, current_position, initial_position, acceptable_profit_rate):
    profit_rate = (current_position - initial_position) / initial_position
    if profit_rate <= 0.0:
        return loss_rate
    return max(min_loss_rate, loss_rate * math.e ** (- profit_rate / acceptable_profit_rate))

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# tools for training and testing

def train_save(agent):
    T.save(agent, agent.file_path)
    print(f"Training saved at {agent.file_path}.")

def train_load(file_path):
    train_agent = T.load(file_path)
    train_agent.train()
    print(f"Training loaded at {file_path}.")
    return train_agent

def inference_load(file_path):
    # with T.no_grad():
    train_agent = T.load(file_path)
    train_agent.eval()
    print(f"Testing loaded at {file_path}.")
    return train_agent

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# pre-made models

def lin_reg_model(data):
    corr = data.corr()
    print(corr["Profit"])
    drop_list = ["Profit"]
    for column in corr.columns[1:]:
        if corr["Profit"][column] < 0.2:
            drop_list.append(column)
    
    X = np.array(data.drop(columns = drop_list))
    y = np.array(data[["Profit"]])

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

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
