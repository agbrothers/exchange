import sys
import random
import datetime
import numpy as np
import pandas as pd
from gym.spaces import Box

from base_env import BaseEnv
from preprocessors import StockPreprocessor


class Market():

    # NOTE: 
    # This environment is open-loop and assumes the quantities traded will not materially impact the price

    def __init__(self, env_config:dict):
        np.random.seed(env_config["seed"])
        self.stocks = {}
        self.calendar = set()
        self.dt = max(env_config["dt"], 1) # in minutes
        self.starting_balance = env_config["starting_balance"]
        self.tickers = env_config["tickers"]
        self.features = ["open","high","low","close","volume"]
        self.order_lag = env_config["order_lag"] # Lag in minutes for buy/sell orders to execute

        self.load_data()
        self.max_steps = self.stocks[self.tickers[0]]["historical_data"].shape[0] // self.dt # Number of minutes from market open to close
        self.norm_constants=np.array([])

    def load_data(self):
        ticker_paths = {
            "AAPL": "AAPL_1min_sample.csv",
            "MSFT": "MSFT_1min_sample.csv",
        }
        now = datetime.datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0) # 9:30 am EST
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0) # 4:00 pm EST

        for i,ticker in enumerate(self.tickers):
            if ticker not in ticker_paths.keys():
                sys.error(f"ERROR: {ticker} is not currently supported")
            data = pd.read_csv(f"data/{ticker_paths[ticker]}")
            data['time'] = pd.to_datetime(data['time'])
            data = data[data["time"] >= market_open]
            data = data[data["time"] < market_close]
            dates = list(data.groupby("date").groups.keys())
            self.stocks[ticker] = {
                "historical_data": data,
                "dates": dates,
            }
            self.calendar=set(dates) if i==0 else self.calendar.intersection(set(dates))


    def reset(self):
        self.current_step = 0
        self.current_portfolio_value = self.starting_balance
        self.current_cash_balance = self.starting_balance
        self.date = np.random.choice(list(self.calendar))
        initial_conditions = []

        for ticker,stock in self.stocks.items():
            self.stocks[ticker]["live_data"] = stock["historical_data"].groupby("date").get_group(self.date)[self.features].values
            self.stocks[ticker]["current_price"] = list(self.stocks[ticker]["live_data"][0])            
            self.stocks[ticker]["preprocessor"] = StockPreprocessor(self.stocks[ticker].get("current_price"), self.dt)
            initial_conditions += list(self.stocks[ticker]["preprocessor"].get_state())  #self.stocks[ticker].get("current_price")
            self.norm_constants = np.append(self.norm_constants, self.stocks[ticker]["preprocessor"].norm_constants)
        return initial_conditions


    def step(self):
        # Add N time step lag for buy/sell orders
        self.current_step += 1 
        obs = []

        for ticker in self.stocks:
            self.stocks[ticker]["current_price"] = self.stocks[ticker].get("live_data")[self.current_step * self.dt]            
            self.stocks[ticker]["preprocessor"].update(self.stocks[ticker].get("current_price"))
            obs+= list(self.stocks[ticker]["preprocessor"].get_state())
        done = self.current_step==self.max_steps-1
        return obs, done


    def action_space(self):
        pass
    def observation_space(self):
        pass
    def close(self):
        pass
    def render(self):
        pass



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
    env = Market(env_config)




        
