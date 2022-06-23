import numpy as np
import threading
import time

from analyze import stock
from gbras import GoalBasedRiskAwareSeller
from simulated_market import SimulatedMarket
from tools import *

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class RiskAwareGreedyBuyer(object):
    def __init__(self):
        self.total_change = 0.0
        self.initial_price = 100.0
        self.current_price = self.initial_price
        self.last_local_high = 0.0
        self.last_local_change = 0.0

        self.buy_stop = 0.0
        self.buy_stop_factor = 0.02

        self.reach_margin = 0.005
        self.top_reach = 0.0
        self.bottom_reach = 0.0

    def prepare(self, current_price = 100.0):
        self.initial_price = current_price
        self.current_price = current_price

        self.buy_stop = self.initial_price * (1 + self.buy_stop_factor)
        self.bottom_reach = self.buy_stop * (1 - self.reach_margin)

    def buy(self, current_price):
        last_price = self.current_price
        self.current_price = current_price
        price_movement = (current_price - last_price) / last_price

        if price_movement < 0:
            self.buy_stop *= (1 + price_movement)
            self.bottom_reach = self.buy_stop * (1 - self.reach_margin)
        if self.current_price >= self.bottom_reach and self.buy_stop != 0.0:
            return True
        return False

data = stock(0)
l = len(data)
market = SimulatedMarket(data)
buyer = RiskAwareGreedyBuyer()
buyer.prepare()
for i in range(l):
    current_price, current_price_movement = market.update()
    print(f"Step = {market.current_step}, Current price = {current_price}, Price Movement = {current_price_movement}")
    
    bought = buyer.buy(current_price)
    if bought:
        buyer.prepare()
        print("Bought")
