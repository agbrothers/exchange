import sys
import time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import copy
from sklearn.model_selection import train_test_split

from envs.preprocessors import StockPreprocessor
from database.database import DataBase


class Market:

    # NOTE: 
    # This environment is open-loop and assumes the quantities traded will not materially impact the price
    CURRENT_PRICE_IDX = 7

    def __init__(self, env_config:dict):

        np.random.seed(env_config["seed"])
        self.DB = DataBase(env_config["database_credentials_path"])
        self.combinatorial_training = env_config["combinatorial_training"]
        self.train_test_split = env_config["train_test_split"]
        self.starting_balance = env_config["starting_balance"]
        self.order_lag = env_config["order_lag"] # Lag in minutes for buy/sell orders to execute
        self.training = env_config["training"]
        self.tickers = env_config["tickers"]
        self.dt = max(env_config["dt"], 1) # in minutes
        # Number of minutes from market open to close  
        self.max_steps = 390 // self.dt 
        self.step_num = None
        self._state = None
        self.date = None
        self.run_num = 0
        self.preprocessor_dict = {}
        self.features = [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
        # If no tickers are specified, use all tickers
        if self.tickers == None:
            query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'exchange' 
                AND table_name NOT LIKE 'tickers';
                """
            df = self.DB.read(query)
            self.tickers = df.TABLE_NAME.values

        # Split train / test data chronologically
        self.calendar = self.DB.read("select distinct date from MSFT")['date'].values
        train_test_split_idx = int(len(self.calendar) * self.train_test_split)
        if self.training:
            dates = self.calendar[:train_test_split_idx]
        else:
            dates = self.calendar[train_test_split_idx:]
        np.random.shuffle(dates)
        self.dates = iter(dates)


    def load_data(self):
        # Load the entire day stack
        # NOTE: This is very slow right now
        values = []
        for ticker in tqdm(copy(self.selected_tickers)):
            query = f"""
            select open, high, low, close, volume from {ticker}
            where date = '{self.date}';
            """
            df = self.DB.read(query)
            if df.empty: 
                self.selected_tickers.remove(ticker)
            else:
                values.append(df[self.features].values)
        return np.stack(values)

    # def load_data(self):
    #     # Load the entire day stack
    #     # NOTE: This is very slow right now
    #     print(f"Loading Data from {self.date}")
    #     query = ""
    #     for i,ticker in enumerate(self.selected_tickers):   
    #         if i != 0: query += " UNION ALL "
    #         query += f"SELECT open, high, low, close, volume FROM {ticker} WHERE date = '{self.date}'"

    #     t0 = time.time()
    #     df = self.DB.read(query)
    #     print(f"Data Loaded: {time.time() - t0}")
    #     return df.values.reshape((-1, self.max_steps*self.dt, len(self.features)))


    def reset(self, eval=False):
        print(f"Reset {self.run_num}")
        self.run_num += 1
        self.step_num = 0
        self.preprocessor_dict = {}
        self.date = next(self.dates)

        if self.combinatorial_training:
            self.selected_tickers = list(np.random.choice(
                self.tickers, 
                np.random.randint(len(self.tickers)),
                replace=False,
            ))
        else:
            self.selected_tickers = self.tickers

        processed_ics = []
        normalized_processed_ics = []
        print(f"Loading Data from {self.date}")
        self.data = self.load_data()
        for i,ticker in enumerate(self.selected_tickers):
            raw_ics = self.data[i, self.step_num, :]
            self.preprocessor_dict[ticker] = StockPreprocessor(raw_ics, self.dt)
            processed_ics.append(self.preprocessor_dict[ticker].get_state())
            normalized_processed_ics.append(self.preprocessor_dict[ticker].get_norm_state())
        self._state = np.stack(processed_ics)

        return np.stack(normalized_processed_ics), self.selected_tickers


    def step(self):
        # TODO: Add N time step lag for buy/sell orders
        #       if the agent gets gets good enough
        self.step_num += 1 
        processed_obs = []
        normalized_processed_obs = []
        for i,ticker in enumerate(self.selected_tickers):
            raw_obs = self.data[i, self.step_num, :]
            self.preprocessor_dict[ticker].update(raw_obs)
            processed_obs.append(self.preprocessor_dict[ticker].get_state())
            normalized_processed_obs.append(self.preprocessor_dict[ticker].get_norm_state())
        self._state = np.stack(processed_obs)

        done = self.step_num >= self.max_steps-1
        return np.stack(normalized_processed_obs), done

    @property
    def state(self):
        return self._state


if __name__ == "__main__":

    env_config = {
        "seed":0,
        "dt":1,
        "order_lag":0,
        "starting_balance":100_000,
        "database_credentials_path": "database/credentials.json",
        "combinatorial_training": True,
        "train_test_split": 0.8,
        "training": True,
        "tickers": None,
    }
    env = Market(env_config)




        
