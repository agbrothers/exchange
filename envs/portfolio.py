
class Portfolio:  # properties and state of physical world entity
    def __init__(self, env_config):

        self.name = ""
        self.starting_balance = env_config["starting_balance"]
        self.tickers = env_config["tickers"]
        # state
        # self.state = EntityState()


    def reset(self):
        self.cash_balance = self.starting_balance
        self.portfolio_value = self.starting_balance
        self.num_positions = 0
        self.positions = {}
        for ticker in self.tickers:
            self.positions[ticker] = {
                "num_shares":0,
                "value":0,
                "portfolio_diversity":0,
            }


    def step(self, trades, market_obs):    
        for trade in trades:
            self.portfolio[trade["ticker"]] += trade.num_shares
            self.cash_balance -= trade.value

        self.prev_portfolio_value = self.portfolio_value
        self.portfolio_value = 0
        for ticker in self.positions:
            price_per_share = market_obs[ticker]["current_price"][3] # 3 ~ step close price
            self.position[ticker]["value"] = price_per_share * self.position[ticker]["num_shares"] 
            self.portfolio_value += self.position[ticker]["value"]

        portfolio_obs = [self.portfolio_value, self.cash_balance]
        
        for ticker in self.positions:
            self.position[ticker]["portfolio_diversity"] = self.position[ticker]["value"] / self.portfolio_value
            portfolio_obs.append([
                self.position[ticker]["num_shares"],
                self.position[ticker]["value"], 
                self.position[ticker]["portfolio_diversity"]
            ])
        step_percentage_change = 1 - (self.portfolio_value / self.prev_portfolio_value)
        day_percentage_change = 1 - (self.portfolio_value / self.starting_balance)   
        reward = step_percentage_change + 0.1 * day_percentage_change
        
        return portfolio_obs, reward