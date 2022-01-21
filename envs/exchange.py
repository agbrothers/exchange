import yaml
import numpy as np
import gym
from gym.spaces import Box

from base_env import BaseEnv
from market import Market
from portfolio import Portfolio


class Exchange(gym.Env):

    # NOTE: 
    # This environment is open-loop and assumes the quantities traded will not materially impact prices

    def __init__(self, env_config:dict):
        self.market = Market(env_config)        # ~Environment 
        self.portfolio = Portfolio(env_config)  # ~Reward Function

        self.state = None
        self.current_step = None
        self.norm_constants=None

        ics = self.reset()
        self.max_steps = self.market.max_steps
        self._action_space = Box(-1, 1, shape=[len(env_config["tickers"])]) # sell -1 > 0, hold == 0, buy 0 < 1        
        self._observation_space = ics.shape     # np.array(2 + 3*self.num_stocks + 28*self.num_stocks)


    def reset(self):
        self.current_step = 0
        market_ics = self.market.reset()
        portfolio_ics = self.portfolio.reset(market_ics)
        self.norm_constants = np.concatenate((self.portfolio.norm_constants, self.market.norm_constants))
        self.state = np.concatenate((portfolio_ics, market_ics))
        return self.state/self.norm_constants


    def step(self, action):
        # Add N time step lag for buy/sell orders
        self.current_step += 1
        market_obs, done = self.market.step()
        portfolio_obs, reward = self.portfolio.step(action, self.market.stocks)
        self.state = np.concatenate((portfolio_obs, market_obs))
        return self.state/self.norm_constants, reward, done


    def action_space(self):
        return self._action_space
    def observation_space(self):
        return self._observation_space
    def close(self):
        pass
    def render(self):
        pass
    def seed(self):
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
    env = Exchange(env_config)
    env.step(np.array([0.5,0.3]))




        
