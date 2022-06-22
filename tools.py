import numpy as np
import math

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