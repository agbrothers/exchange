from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import gym
import torch
import numpy as np
from collections import OrderedDict
# from gym.spaces import Box, MultiDiscrete
from numpy.core.fromnumeric import clip

from envs.market import Market
from envs.spaces import MutableBox, MutableMultiDiscrete

class Exchange(gym.Env):

    # NOTE: 
    # This environment is open-loop and assumes the quantities traded will not materially impact prices

    def __init__(self, env_config:dict):
        self.starting_balance = self.portfolio_value = self.cash_balance = env_config["starting_balance"]

        self.market = Market(env_config)        # ~Environment 
        self.portfolio = None 
        self.current_step = None
        self.selected_tickers = None
        self.buy_sell_percentage = 0.10 # tune this to see how it impacts learning
        self.max_daily_increase_multiple = 10 # For scaling, don't expect more than 10x portfolio value increase in a day
        self.max_steps = self.market.max_steps
        self.action_space = MutableMultiDiscrete([2, 2, 10] * len(self.market.tickers))
        self.observation_space =  MutableBox(-10, 10, shape=(len(self.market.tickers)*29 + 2,))
        self.debug = env_config["debug"]


    def update_spaces(self):
        num_stocks = len(self.selected_tickers)
        self.action_space.update([2, 2, 10] * num_stocks)
        self.observation_space.update(shape=(num_stocks*29 + 2,))

    def reset(self, eval=False):
        self.current_step = 0
        self.portfolio_value = self.cash_balance = self.starting_balance
        market_ics, self.selected_tickers = self.market.reset(eval)

        self.portfolio = OrderedDict([(ticker, 0.0) for ticker in self.selected_tickers])
        portfolio_ics = np.zeros((len(self.selected_tickers),1))
        combined_ics = np.concatenate((market_ics, portfolio_ics), axis=1)
        flattened_ics = np.concatenate(combined_ics[:])

        rel_portfolio_value = self.portfolio_value / self.starting_balance / self.max_daily_increase_multiple 
        rel_cash_balance = self.cash_balance / self.starting_balance / self.max_daily_increase_multiple 
        ics = np.append([rel_portfolio_value, rel_cash_balance], flattened_ics)

        self.update_spaces()
        if self.debug: print(self.current_step, int(self.portfolio_value))
        return ics


    def step(self, action):
        # Add N time step lag for buy/sell orders
        # print(action, self.portfolio.cash_balance, self.portfolio.positions["AAPL"]["portfolio_diversity"], self.portfolio.positions["MSFT"]["portfolio_diversity"])
        self.current_step += 1
        reward = 0
        prices = self.market.state
        if not isinstance(action, np.ndarray) and not isinstance(action, torch.Tensor):
            action = np.array(action)
        actions_per_stock = action.reshape((-1, 3))

        # TAKE ACTIONS
        for i,ticker in enumerate(self.selected_tickers):
            a = actions_per_stock[i]
            # a[0] : 0 ~ act,  1 ~ hold
            # a[1] : 0 ~ sell, 1 ~ buy
            # a[2] : 1 - 10, how much stock to buy/sell relative to starting balance
            assert a.shape == (3,)

            # Check if hold
            if a[0] == 1:
                continue

            value_to_move  = a[2] * self.buy_sell_percentage * self.starting_balance
            shares_to_move = value_to_move / prices[i][Market.CURRENT_PRICE_IDX]

            # Check if sell
            if a[1] == 0:
                if self.portfolio[ticker] >= shares_to_move:
                    self.portfolio[ticker] -= shares_to_move
                    self.cash_balance += value_to_move
                    if self.debug: print(f"- Sold {value_to_move} of {ticker}")

            # Check if buy
            elif a[1] == 1:
                if self.cash_balance >= value_to_move:
                    self.cash_balance -= value_to_move
                    self.portfolio[ticker] += shares_to_move
                    if self.debug: print(f"+ Bought {value_to_move} of {ticker}")


        # UPDATE MARKET & PORTFOLIO
        market_obs, done = self.market.step()
        portfolio_obs = self.get_portfolio_values(self.market.state)
        self.portfolio_value = np.sum(portfolio_obs) + self.cash_balance

        normalized_portfolio_obs = portfolio_obs / self.starting_balance
        combined_obs = np.concatenate((market_obs, normalized_portfolio_obs), axis=1)
        flattened_obs = np.concatenate(combined_obs[:])

        rel_portfolio_value = self.portfolio_value / self.starting_balance / self.max_daily_increase_multiple 
        rel_cash_balance = self.cash_balance / self.starting_balance / self.max_daily_increase_multiple 
        obs = np.append([rel_portfolio_value, rel_cash_balance], flattened_obs)
        clipped_obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        # Terminal reward is the daily % change
        if self.debug: print(self.current_step, int(self.portfolio_value))
        if done: 
            reward += self.portfolio_value / self.starting_balance
            print(f"FINAL PORTFOLIO VALUE: {int(self.portfolio_value)}, REWARD: {round(reward, 3)}")
        return clipped_obs, reward, done, {}


    def get_portfolio_values(self, prices):
        values = []
        for i,ticker in enumerate(self.selected_tickers):
            current_price = prices[i][Market.CURRENT_PRICE_IDX]
            values.append(self.portfolio[ticker] * current_price)

        portfolio_obs = np.array(values).reshape((-1,1))
        return portfolio_obs


    # @property
    # def observation_space(self):
    #     # [ 2 + (num_stocks * state_size=29) ]
    #     return self._observation_space

    # @property
    # def action_space(self):
    #     # [ num_stocks, action_size=3 ]
    #     return self._action_space

    def close(self):
        pass

    def render(self):
        pass

    def seed(self):
        super().seed()


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

    env = Exchange(env_config)

    actions = [0, 1, 2] * len(env.selected_tickers)
    env.step(np.array(actions))

    actions = [0, 0, 1] * len(env.selected_tickers)
    env.step(np.array(actions))
    env


        
