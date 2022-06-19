import numpy as np
import pandas as pd
from collections import deque
import itertools

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# assume data type is pandas series
class Indicators(object):
    def __init__(self, capacity = 10000, periods = [14, 30, 100]):
        self.n_periods = 3
        self.periods = periods

        # second dim: {0: pos, 1: value}
        self.highest_in_period = [[0, 0.0] for _ in range(self.n_periods)]
        self.lowest_in_period = [[0, 0.0] for _ in range(self.n_periods)]

        # 0: time, 1: open, 2: high, 3: low, 4: close, 5: volume
        self.capacity = capacity
        self.market_data = deque([], maxlen = capacity)
        self.list = []
        for i in range(self.n_periods):
            self.list.append("SMA {}".format(self.periods[i]))
            self.list.append("WMA {}".format(self.periods[i]))
            self.list.append("EMA {}".format(self.periods[i]))
        self.list.extend(["Aroon Up", "Aroon Down", "MACD", "Momentum", "Stoch K", "Stoch D", "William R", "AD"])
        self.data = {
            "SMA {}".format(self.periods[0]): deque(maxlen = capacity), 
            "WMA {}".format(self.periods[0]): deque(maxlen = capacity), 
            "EMA {}".format(self.periods[0]): deque(maxlen = capacity), 
            "SMA {}".format(self.periods[1]): deque(maxlen = capacity), 
            "WMA {}".format(self.periods[1]): deque(maxlen = capacity), 
            "EMA {}".format(self.periods[1]): deque(maxlen = capacity), 
            "SMA {}".format(self.periods[2]): deque(maxlen = capacity), 
            "WMA {}".format(self.periods[2]): deque(maxlen = capacity), 
            "EMA {}".format(self.periods[2]): deque(maxlen = capacity), 
            "Aroon Up": deque(maxlen = capacity), 
            "Aroon Down": deque(maxlen = capacity), 
            "MACD": deque(maxlen = capacity), 
            "Momentum": deque(maxlen = capacity), 
            "Stoch K": deque(maxlen = capacity), 
            "Stoch D": deque(maxlen = capacity), 
            "William R": deque(maxlen = capacity), 
            "AD": deque(maxlen = capacity)
        }

        self.current_price = 0.0
        self.memory_count = 0

    def update(self, time, open, high, low, close, volume):
        self.current_price = close
        self.market_data.append([time, open, high, low, close, volume])

        self.highest_in_period = [self.__recent_highest__(self.periods[i]) for i in range(self.n_periods)]
        self.lowest_in_period = [self.__recent_lowest__(self.periods[i]) for i in range(self.n_periods)]

        if self.memory_count > 0:
            for i in range(self.n_periods):
                self.data["SMA {}".format(self.periods[i])].append(self.__sma__(self.periods[i], self.data["SMA {}".format(self.periods[i])][-1], close))
                self.data["WMA {}".format(self.periods[i])].append(self.__wma__(self.periods[i], self.data["WMA {}".format(self.periods[i])][-1], close))
                self.data["EMA {}".format(self.periods[i])].append(self.__ema__(self.periods[i], self.data["EMA {}".format(self.periods[i])][-1], close))
        else:
            for i in range(self.n_periods):
                self.data["SMA {}".format(self.periods[i])].append(close)
                self.data["WMA {}".format(self.periods[i])].append(close)
                self.data["EMA {}".format(self.periods[i])].append(close)
        self.data["Aroon Up"].append(self.__aroon_up__(self.periods[0], self.memory_count - self.highest_in_period[0][0]))
        self.data["Aroon Down"].append(self.__aroon_down__(self.periods[0], self.memory_count - self.lowest_in_period[0][0]))
        self.data["MACD"].append(self.__macd__())
        self.data["Momentum"].append(self.__momentum__(open, close))
        self.data["Stoch K"].append(self.__stoch_k__(self.highest_in_period[0][1], self.lowest_in_period[0][1], self.current_price))
        self.data["Stoch D"].append(self.__stoch_d__(self.periods[0], self.data["Stoch K"][-1], self.highest_in_period[0][1], self.lowest_in_period[0][1], self.current_price))
        self.data["William R"].append(self.__william_r__(self.highest_in_period[0][1], self.lowest_in_period[0][1], self.current_price))
        self.data["AD"].append(self.__last_ad__(self.market_data[-1][2], self.market_data[-1][3], self.market_data[-1][4]))

        self.memory_count += 1

    def __recent_highest__(self, period):
        period_data = list(itertools.islice(self.market_data, max(0, self.memory_count - period + 1), self.memory_count + 1))
        highest = period_data[0][2]
        begin = max(0, self.memory_count - period + 1)
        index = max(0, self.memory_count - period + 1)
        for i in range(1, len(period_data)):
            if highest <= period_data[i][3]:
                index = begin + i
                highest = period_data[i][3]
        return [index, highest]

    def __recent_lowest__(self, period):
        period_data = list(itertools.islice(self.market_data, max(0, self.memory_count - period + 1), self.memory_count + 1))
        lowest = period_data[0][3]
        begin = max(0, self.memory_count - period + 1)
        index = max(0, self.memory_count - period + 1)
        for i in range(1, len(period_data)):
            if lowest >= period_data[i][3]:
                index = begin + i
                lowest = period_data[i][3]
        return [index, lowest]

    def __sma__(self, period, last_sma, current_price):
        return (last_sma * (period - 1) + current_price) / period

    def __wma__(self, period, last_wma, current_price):
        sum_weights = period * (period + 1) / 2 
        return (last_wma * (sum_weights - period) + period * current_price) / sum_weights

    def __ema__(self, period, last_ema, current_price):
        return last_ema * (1 - 2 / (period + 1)) + current_price * (2 / (period + 1))

    def __aroon_up__(self, period, highest_point_dist):
        return 1 - highest_point_dist / period

    def __aroon_down__(self, period, lowest_point_dist):
        return 1 - lowest_point_dist / period

    def __macd__(self):
        return self.data["EMA {}".format(self.periods[0])][-1] - self.data["EMA {}".format(self.periods[1])][-1]

    # def signal(self, last_signal, current_price):
    #     return self.macd()

    def __momentum__(self, start_price, end_price):
        return end_price - start_price

    def __stoch_k__(self, highest_in_period, lowest_in_period, current_price):
        return (current_price - lowest_in_period) / (highest_in_period - lowest_in_period) if highest_in_period != lowest_in_period else 0.0

    def __stoch_d__(self, period, last_stoch_k, highest_in_period, lowest_in_period, current_price):
        return (last_stoch_k * (period - 1) + self.__stoch_k__(highest_in_period, lowest_in_period, current_price) / period)

    def __william_r__(self, highest_in_period, lowest_in_period, current_price):
        return (highest_in_period - current_price) / (highest_in_period - lowest_in_period) if highest_in_period != lowest_in_period else 0.0

    def __last_ad__(self, last_high, last_low, last_price):
        return (last_high - last_price) / (last_high - last_low) if last_high != last_low else 0.0

    # def last_cci(self, period_data):
    #     period = 14     # = len(period_data)
    #     m = [(period_data[i][1] + period_data[i][2] + period_data[i][3]) / 3 for i in range(period)]
    #     sm = sum(m)
    #     for i in range(period):
    #         m[i] = abs(m[i] - sm)
    #     return sum(m) / period

