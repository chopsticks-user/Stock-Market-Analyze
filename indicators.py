import numpy as np
import pandas as pd
from collections import deque
import itertools

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# assume data type is pandas series
class Indicators(object):
    def __init__(self, capacity = 10000, periods = [14, 30, 100]):
        self.n_periods = len(periods)
        self.periods = periods
        self.last_sma = [{"SMA {}".format(periods[i]): 0.0} for i in range(self.n_periods)]
        self.last_wma = [{"WMA {}".format(periods[i]): 0.0} for i in range(self.n_periods)]
        self.last_ema = [{"EMA {}".format(periods[i]): 0.0} for i in range(self.n_periods)]

        # second dim: {0: pos, 1: value}
        self.highest_in_period = [[0, 0.0] for _ in range(self.n_periods)]
        self.lowest_in_period = [[0, 0.0] for _ in range(self.n_periods)]

        # 0: time, 1: open, 2: high, 3: low, 4: close, 5: volume
        self.capacity = capacity
        self.market_data = deque([], maxlen = capacity)
        self.indicators_data = deque([], maxlen = capacity)
        self.current_price = 0.0
        self.memory_count = 0

    def update(self, time, open, high, low, close, volume):
        self.current_price = close
        self.market_data.append([time, open, high, low, close, volume])
        if self.memory_count == 0:
            self.last_sma = [{"SMA {}".format(self.periods[i]): close} for i in range(self.n_periods)]
            self.last_wma = [{"WMA {}".format(self.periods[i]): close} for i in range(self.n_periods)]
            self.last_ema = [{"EMA {}".format(self.periods[i]): close} for i in range(self.n_periods)]
            self.highest_in_period = [[0, high] for _ in range(self.n_periods)]
            self.lowest_in_period = [[0, low] for _ in range(self.n_periods)]
        else:
            self.highest_in_period = [self.recent_highest(self.periods[i]) for i in range(self.n_periods)]
            self.lowest_in_period = [self.recent_lowest(self.periods[i]) for i in range(self.n_periods)]

        self.last_sma = [self.sma(self.periods[i], self.last_sma[i]["SMA {}".format(self.periods[i])], close) for i in range(self.n_periods)]
        self.last_wma = [self.wma(self.periods[i], self.last_wma[i]["WMA {}".format(self.periods[i])], close) for i in range(self.n_periods)]
        self.last_ema = [self.ema(self.periods[i], self.last_ema[i]["EMA {}".format(self.periods[i])], close) for i in range(self.n_periods)]
        aroon = self.aroon(self.periods[0], self.memory_count - self.lowest_in_period[0][0], self.memory_count - self.highest_in_period[0][0])
        macd = self.macd()
        momentum = self.momentum(open, close)
        stoch_k = self.stoch_k(self.highest_in_period[0][1], self.lowest_in_period[0][1], self.current_price)
        stoch_d = self.stoch_d(self.periods[0], stoch_k, self.highest_in_period[0][1], self.lowest_in_period[0][1], self.current_price)
        william_r = self.william_r(self.highest_in_period[0][1], self.lowest_in_period[0][1], self.current_price)
        last_ad = self.last_ad(self.market_data[-1][2], self.market_data[-1][3], self.market_data[-1][4])

        self.memory_count += 1
        self.indicators_data.append([*self.last_sma, *self.last_wma, *self.last_ema, *aroon, macd, momentum, stoch_k, stoch_d, william_r, last_ad])
        return *self.last_sma, *self.last_wma, *self.last_ema, *aroon, macd, momentum, stoch_k, stoch_d, william_r, last_ad

    def recent_highest(self, period):
        period_data = list(itertools.islice(self.market_data, max(0, self.memory_count - period + 1), self.memory_count + 1))
        highest = period_data[0][2]
        begin = max(0, self.memory_count - period + 1)
        index = max(0, self.memory_count - period + 1)
        for i in range(1, len(period_data)):
            if highest <= period_data[i][3]:
                index = begin + i
                highest = period_data[i][3]
        return [index, highest]

    def recent_lowest(self, period):
        period_data = list(itertools.islice(self.market_data, max(0, self.memory_count - period + 1), self.memory_count + 1))
        lowest = period_data[0][3]
        begin = max(0, self.memory_count - period + 1)
        index = max(0, self.memory_count - period + 1)
        for i in range(1, len(period_data)):
            if lowest >= period_data[i][3]:
                index = begin + i
                lowest = period_data[i][3]
        return [index, lowest]

    def sma(self, period, last_sma, current_price):
        return {"SMA {}".format(period): (last_sma * (period - 1) + current_price) / period}

    def wma(self, period, last_wma, current_price):
        sum_weights = period * (period + 1) / 2 
        return {"WMA {}".format(period): (last_wma * (sum_weights - period) + period * current_price) / sum_weights}

    def ema(self, period, last_ema, current_price):
        # weight = 2 / (period + 1)
        return {"EMA {}".format(period): last_ema * (1 - 2 / (period + 1)) + current_price * (2 / (period + 1))}

    def aroon(self, period, lowest_point_dist, highest_point_dist):
        aroon_up = 1 - highest_point_dist / period
        aroon_down = 1 - lowest_point_dist / period
        return {"Aroon Up": aroon_up}, {"Aroon Down": aroon_down}

    def macd(self):
        macd = self.last_ema[0]["EMA {}".format(self.periods[0])] - self.last_ema[1]["EMA {}".format(self.periods[1])]
        return {"MACD": macd}

    # def signal(self, last_signal, current_price):
    #     return self.macd()

    def momentum(self, start_price, end_price):
        return {"Momentum": end_price - start_price}

    def stoch_k(self, highest_in_period, lowest_in_period, current_price):
        return {"Stoch K": (current_price - lowest_in_period) / (highest_in_period - lowest_in_period)}

    def stoch_d(self, period, last_stoch_k, highest_in_period, lowest_in_period, current_price):
        return {"Stoch D": (last_stoch_k["Stoch K"] * (period - 1) + self.stoch_k(highest_in_period, lowest_in_period, current_price)["Stoch K"] / period)}

    def william_r(self, highest_in_period, lowest_in_period, current_price):
        return {"William R": (highest_in_period - current_price) / (highest_in_period - lowest_in_period)}

    def last_ad(self, last_high, last_low, last_price):
        return {"Last AD": (last_high - last_price) / (last_high - last_low)}

    # def last_cci(self, period_data):
    #     period = 14     # = len(period_data)
    #     m = [(period_data[i][1] + period_data[i][2] + period_data[i][3]) / 3 for i in range(period)]
    #     sm = sum(m)
    #     for i in range(period):
    #         m[i] = abs(m[i] - sm)
    #     return sum(m) / period

