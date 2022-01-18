import sys
import yaml
import random
import datetime
import numpy as np
import pandas as pd
from gym.spaces import Box
from envs.preprocessors import BaseAccount

from base_env import BaseEnv
from envs.preprocessors import Portfolio, StockPreprocessor

import coax
import pettingzoo
# An environment that simulates a day of trading on 1min, 5min, 30min, and 1 hour resolutions

class Exchange(BaseEnv):

    # NOTE: 
    # This environment is open-loop and assumes the quantities traded will not materially impact the price

    def __init__(self, env_config:dict):

        np.random.seed(env_config["seed"])
        self.stocks = {}
        self.calendar = set()

        # Entities
        self.portfolio = Portfolio()
        self.trader = Trader()

        # GET CONFIG PARAMETERS    
        self.dt = max(env_config["dt"], 1) # in minutes
        self.order_lag = env_config["order_lag"] # Lag in minutes for buy/sell orders to execute
        self.starting_balance = env_config["starting_balance"]


        # BUILD TRADING ENVIRONMENT
        self.tickers = env_config["tickers"]
        self.features = ["open","high","low","close","volume"]
        self.load_data()
        self.max_steps = self.stocks[self.tickers[0]]["data"].shape[0] // self.dt # Number of minutes from market open to close
        

        # NOTE: Could have continuous action space for buying fractional shares
        # How will buying/selling work with discrete actions?
            # Should have continuous action spaces [-1,1] buy / sell that % of the balance or value
            # Will need a action for each available stock
        # self.actions = Box(0, np.inf, shape=(3)) # buy $N, sell $N, hold
        # self.obs = Box(-np.inf, np.inf, shape=(6)) # open, high, low, close, volume, balance, % change

        self.reset()



    def load_data(self):

        ticker_paths = {
            "AAPL": "AAPL_1min_sample.csv",
            "MSFT": "MSFT_1min_sample.csv",
        }

        now = datetime.datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0) # Includes pre-market
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0) # Includes after-hours

        for i,ticker in enumerate(self.tickers):
            if ticker not in ticker_paths.keys():
                sys.error(f"ERROR: {ticker} is not currently supported")

            data = pd.read_csv(f"data/{ticker_paths[ticker]}")
            data['time'] = pd.to_datetime(data['time'])
            data = data[data["time"] >= market_open]
            data = data[data["time"] < market_close]

            dates = list(data.groupby("date").groups.keys())
            preprocessor = StockPreprocessor()

            self.stocks[ticker] = {
                "historical_data": data,
                "dates": dates,
                "preprocessor": preprocessor,
            }
            self.calendar=set(dates) if i==0 else self.calendar.intersection(set(dates))


    def seed(self):
        samples: int = self.data.shape[0]
        starting_points: int = samples - self.time_steps - 1
        random.randint(0, starting_points)


    def reset(self):
        self.current_step = 0
        self.current_portfolio_value = self.starting_balance
        self.current_cash_balance = self.starting_balance

        self.actions = Box(-1, 1, shape=[len(self.tickers)]) # sell -1 > 0, hold == 0, buy 0 < 1
        market_initial_conditions = []

        self.date = np.random.choice(list(self.calendar))

        for ticker,stock in self.stocks.items():
            self.stocks[ticker]["live_data"] = stock["data"].groupby("date").get_group(self.date)[self.features].values
            self.stocks[ticker]["current_price"] = self.stocks[ticker]["live_data"][0]            
            market_initial_conditions.append(self.stocks[ticker]["current_price"])
        

        portfolio_initial_conditions = self.portfolio.reset()
        initial_conditions = portfolio_initial_conditions + market_initial_conditions
        return initial_conditions



    def step(self, actions):
        # Add N time step lag for buy/sell orders
        self.current_step += 1
        self.obs = []

        for ticker in self.stocks:
            self.obs.append(self.stocks[ticker]["live_data"][self.current_step])
        
        return self.obs



    def action_space(self, action):
        return super().action_space(action)

    def observation_space(self, action):
        return self.obs

    def close(self, action):
        return super().close(action)

    def render(self, action):
        return super().render(action)



if __name__ == "__main__":


    env_config = {
        "seed":0,
        "dt":1,
        "order_lag":0,
        "starting_balance":100_000,
        "tickers":[
            "AAPL",
            "MSFT",
        ],
    }
    env = Exchange(env_config)




        
