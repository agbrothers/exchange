import numpy as np
import gym
from gym.spaces import Box
from numpy.core.fromnumeric import clip

from envs.market import Market
from envs.portfolio import Portfolio


class Exchange(gym.Env):

    # NOTE: 
    # This environment is open-loop and assumes the quantities traded will not materially impact prices

    def __init__(self, env_config:dict):
        self.market = Market(env_config)        # ~Environment 
        self.portfolio = Portfolio(env_config)  # ~Reward Processor

        self.state = None
        self.current_step = None
        self.norm_constants=None

        ics = self.reset()
        self.max_steps = self.market.max_steps
        self.action_space = Box(-1, 1, shape=[len(env_config["tickers"])]) # sell -1 > 0, hold == 0, buy 0 < 1        
        self.observation_space = Box(-10, 10, shape=[ics.shape[0]])  # Discrete(ics.shape[0])     # np.array(2 + 3*self.num_stocks + 28*self.num_stocks)


    def reset(self, eval=False):
        self.current_step = 0
        market_ics = self.market.reset(eval)
        portfolio_ics = self.portfolio.reset(market_ics, eval)
        print(int(self.portfolio.portfolio_value))
        self.norm_constants = np.concatenate((self.portfolio.norm_constants, self.market.norm_constants))
        self.state = np.concatenate((portfolio_ics, market_ics))
        ics = self.state/self.norm_constants
        return ics


    def step(self, action):
        # Add N time step lag for buy/sell orders
        # print(action, self.portfolio.cash_balance, self.portfolio.positions["AAPL"]["portfolio_diversity"], self.portfolio.positions["MSFT"]["portfolio_diversity"])
        self.current_step += 1
        market_obs, done = self.market.step()
        portfolio_obs, reward = self.portfolio.step(action, self.market.stocks)
        self.state = np.concatenate((portfolio_obs, market_obs))

        # clip observations, assume no more than 100x % change from initial conditions
        obs = self.state/self.norm_constants
        clipped_obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        # Terminal reward is the daily % change
        # print(self.current_step, int(self.portfolio.portfolio_value), reward)
        if done: reward = (self.portfolio.portfolio_value/self.portfolio.starting_balance - 1) * 100
        if done: print(int(self.portfolio.portfolio_value), round(reward,4))
        return clipped_obs, reward, done, {}


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
        "tickers":[
            "AAPL",
            "MSFT",
        ],
    }
    env = Exchange(env_config)
    env.step(np.array([0.5,1]))
    env.step(np.array([-0.5,-1]))
    env



        
