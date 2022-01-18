import yaml
import numpy as np

from envs.base_env import BaseEnv
from envs.market import Market
from envs.trader import Trader
from envs.portfolio import Portfolio


class Exchange(BaseEnv):

    # NOTE: 
    # This environment is open-loop and assumes the quantities traded will not materially impact prices

    def __init__(self, env_config:dict):
        self.market = Market(env_config)        # Environment 
        self.portfolio = Portfolio(env_config)  # Reward Function
        self.trader = Trader(env_config)                  # Agent

        self.state = None
        self.current_step = None

        self.reset()
        self.max_steps = self.market.max_steps


    def reset(self):
        self.current_step = 0
        self.trader.reset()
        market_ics = self.market.reset()
        portfolio_ics = self.portfolio.reset()
        self.state = np.array(portfolio_ics + market_ics,  dtype=np.float32)
        return self.state


    def step(self, action):
        # Add N time step lag for buy/sell orders
        self.current_step += 1
        trades = self.trader.step(action)
        market_obs, done = self.market.step()
        portfolio_obs, reward = self.portfolio.step(trades, market_obs)
        self.state = np.array(portfolio_obs + market_obs,  dtype=np.float32)
        return self.state, reward, done


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
    env = Exchange(env_config)




        
