import numpy as np
import time
from math import e

def random_price_movement():
    return np.tanh(np.random.rand() - 0.45) / 10

def position(pre_position, price_movement):
    return pre_position * (1 + price_movement)

def trailing_stop_loss(current_position, last_stop_loss, inital_stop_loss, loss_rate):
    return max(current_position * (1 - loss_rate), inital_stop_loss, last_stop_loss)

def trigger_stop_loss(stop_loss, current_position, acceptable_margin):
    return current_position <= stop_loss or abs(current_position - stop_loss) <= acceptable_margin * stop_loss

def adjust_loss_rate(loss_rate, current_position, initial_position, acceptable_profit_rate):
    profit_rate = (current_position - initial_position) / initial_position
    if profit_rate <= 0.0:
        return loss_rate
    return max(min_loss_rate, loss_rate * e ** (- profit_rate / acceptable_profit_rate))

transaction_fee_rate = 0.003
initial_capital = 1000.0
capital = initial_capital
acceptable_profit_rate = 0.04
for i in range(365):
    current_month = min(i / 30 + 1, 12)
    for j in range(10):
        initial_position = min(capital, capital) * (1 - transaction_fee_rate)
        current_position = initial_position
        capital -= initial_position / (1 - transaction_fee_rate)
        loss_rate = 0.02
        min_loss_rate = loss_rate * 0.1
        acceptable_margin = 0.005
        initial_stop_loss = current_position * (1 - loss_rate)
        stop_loss = initial_stop_loss

        for k in range(1000):
            price_movement = random_price_movement()
            current_position = position(current_position, price_movement)

            # print(f"Price Movement: {price_movement}, Position: {current_position}, Stop Loss: {stop_loss}")
            if trigger_stop_loss(stop_loss, current_position, acceptable_margin):
                capital += current_position * (1 - transaction_fee_rate)
                # print(capital)
                break

            stop_loss = trailing_stop_loss(current_position, stop_loss, initial_stop_loss, adjust_loss_rate(loss_rate, current_position, initial_position, acceptable_profit_rate))
            # time.sleep(0.5)

        if capital >= initial_capital * (1 + acceptable_profit_rate):
            break

    print(f"Date {i + 1}, Capital: {capital}")