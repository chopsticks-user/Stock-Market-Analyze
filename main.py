from SellAlgorithms.gbras import GoalBasedRiskAwareSeller
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
if __name__ == "__main__":
    seller = GoalBasedRiskAwareSeller()
    for i in range(365): 
        for j in range(5): 
            seller.prepare() 
            stock_range = 1000 
            sold = False 
            for k in range(stock_range): 
                if seller.sell(random_market = True): 
                    break 
            if seller.target_achieved(): 
                break 
        print(f"Date {i + 1}, Capital: {seller.capital}")

