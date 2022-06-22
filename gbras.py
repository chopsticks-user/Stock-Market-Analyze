import numpy as np
import time
import math
from tools import *

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class GoalBasedRiskAwareSeller(object):
    def __init__(self, initial_capital = 1000.0, acceptable_profit_rate = 0.04, loss_rate = 0.02, min_loss_rate_factor = 0.1, 
    loss_acceptable_margin = 0.005, transaction_fee_rate = 0.003):

        self.initial_capital = initial_capital
        self.capital = initial_capital

        self.acceptable_profit_rate = acceptable_profit_rate

        self.loss_rate = loss_rate
        self.min_loss_rate = loss_rate * min_loss_rate_factor
        self.min_loss_rate_factor = min_loss_rate_factor
        self.loss_acceptable_margin = loss_acceptable_margin

        self.initial_stop_loss = 0.0
        self.stop_loss = 0.0
        self.adjust_loss_rate = 0.0

        self.initial_position = 0.0
        self.current_position = 0.0
        self.transaction_fee_rate = transaction_fee_rate

        self.n_trades_made = 0

    def sell(self, current_price_change = 0.0, most_current_change = False, random_market = False):
        price_movement = random_price_movement() if random_market else current_price_change
        self.current_position = position(self.current_position, price_movement)
        if trigger_stop_loss(self.stop_loss, self.current_position, self.loss_acceptable_margin) or most_current_change:
            self.capital += self.current_position * (1 - self.transaction_fee_rate)
            self.n_trades_made += 1
            return True
        self.adjusted_loss_rate = adjust_loss_rate(self.loss_rate, self.min_loss_rate, self.current_position, self.initial_position, self.acceptable_profit_rate)
        self.stop_loss = trailing_stop_loss(self.current_position, self.stop_loss, self.initial_stop_loss, self.adjusted_loss_rate)
        return False

    def prepare(self, initial_position = 0.0):
        if initial_position == 0.0:
            self.initial_position = self.capital
        self.initial_position = min(self.initial_position, self.capital) * (1 - self.transaction_fee_rate)
        self.current_position = self.initial_position
        self.capital -= self.initial_position / (1 - self.transaction_fee_rate)
        self.min_loss_rate = self.loss_rate * 0.1
        self.initial_stop_loss = self.current_position * (1 - self.loss_rate)
        self.stop_loss = self.initial_stop_loss

    def target_achieved(self, greedy = False):
        if not greedy and self.capital >= self.initial_capital * (1 + self.acceptable_profit_rate):
            return True
        return False

    def current_state(self):
        return

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


