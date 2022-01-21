import numpy as np


class Portfolio:  # properties and state of physical world entity
    def __init__(self, env_config):
        self.starting_balance = env_config["starting_balance"]
        self.tickers = env_config["tickers"]
        self.norm_constants = None


    def reset(self, ics):
        self.cash_balance = self.starting_balance
        self.portfolio_value = self.starting_balance
        self.norm_constants = [self.starting_balance, self.starting_balance]
        # reset trading positions and normalization constants
        self.positions = {}
        for i,ticker in enumerate(self.tickers):
            self.positions[ticker] = {
                "num_shares":0,
                "price_per_share": ics[7 + i*28],
                "value":0,
                "portfolio_diversity":0,
            }
            self.norm_constants.append( self.starting_balance / self.positions[ticker].get("price_per_share") )
            self.norm_constants.append( self.starting_balance )
            self.norm_constants.append( 1.0 )

        self.norm_constants = np.array(self.norm_constants)
        portfolio_ics = np.array([self.portfolio_value, self.cash_balance] + [0] * 3 * len(self.tickers), dtype=np.float32)
        return portfolio_ics


    def step(self, action, market_obs): 
        # CONVERT ACTIONS TO TRADES   
        for i,trade in enumerate(action):
            ticker = self.tickers[i]
            if trade == 0: # Hold
                continue
            elif trade > 0: # Buy: action * remaining_cash_bal worth of shares
                buy_order_value = trade * self.cash_balance 
                buy_order_shares = (buy_order_value / self.positions[ticker].get("price_per_share"))
                self.cash_balance -= buy_order_value
                self.positions[ticker]["num_shares"] += buy_order_shares
            else: # Sell: action * stock_value worth of shares
                sell_order_value = trade * self.positions[ticker]["value"]
                sell_order_shares = trade * self.positions[ticker]["num_shares"]
                self.cash_balance -= sell_order_value
                self.positions[ticker]["num_shares"] += sell_order_shares                

        # UPDATE VALUE OF POSITIONS
        self.prev_portfolio_value = self.portfolio_value
        self.portfolio_value = self.cash_balance
        for ticker in self.positions:
            price_per_share = market_obs[ticker].get("current_price")[3] # 3 ~ step close price
            self.positions[ticker]["value"] = price_per_share * self.positions[ticker].get("num_shares")
            self.positions[ticker]["price_per_share"] = price_per_share
            self.portfolio_value += self.positions[ticker].get("value")

        portfolio_obs = [self.portfolio_value, self.cash_balance]
        
        # UPDATE PORTFOLIO DIVERSITY and OBS
        for ticker in self.positions:
            self.positions[ticker]["portfolio_diversity"] = self.positions[ticker]["value"] / self.portfolio_value
            portfolio_obs += [
                self.positions[ticker]["num_shares"],
                self.positions[ticker]["value"], 
                self.positions[ticker]["portfolio_diversity"]
            ]
        portfolio_obs

        # COMPUTE REWARD
        step_percentage_change = 1 - (self.portfolio_value / self.prev_portfolio_value)
        day_percentage_change = 1 - (self.portfolio_value / self.starting_balance)   
        reward = step_percentage_change + 0.1 * day_percentage_change
        return portfolio_obs, reward