import numpy as np
import threading
import time

from analyze import stock
from gbras import GoalBasedRiskAwareSeller
from ragb import RiskAwareGreedyBuyer
from simulated_market import SimulatedMarket
from tools import *

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
data = stock(0)
l = len(data)
market = SimulatedMarket(data)
seller = GoalBasedRiskAwareSeller()
seller.prepare()
for i in range(l - 4300):
    current_price, current_price_movement = market.update()
    print(f"Step = {market.current_step}, Current price = {current_price}, Price Movement = {current_price_movement}")
    
    if seller.sell(current_price_movement):
        if seller.target_achieved():
            print(f"Capital = {seller.capital}")
        seller.prepare()

seller.capital += seller.current_position
print(seller.capital)
