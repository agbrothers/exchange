from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import numpy as np

from envs.exchange import Exchange


class CoreLoop:

    def __init__(self, training_config, env_config):

        self.training = training_config["training"]
        self.rounds = training_config["rounds"]
        self.render = training_config["render"]
        self.agent = training_config["agent"]

        env_config["training"] = self.training
        self.env = Exchange(env_config)

    def run(self):
        for round in range(self.rounds):
            print(f"ROUND {round}")
            obs = self.env.reset()
            done = False
            while not done:
                action = self.agent.predict(obs)
                obs, reward, done, _ = self.env.step(action)

        print("Training Complete")
            
        
class RandomTrader:
    
    def predict(obs):
        num_stocks = len(obs[2:]) // 3
        action = [ np.concatenate((np.random.randint(0,1+1,2), np.random.randint(1,10+1,1)))  for _ in range(num_stocks) ]
        return np.concatenate(action)



if __name__ == "__main__":

    training_config = {
        "rounds": 100,
        "training": True,
        "render": False,
        "agent": RandomTrader,
    }

    env_config = {
        "seed": 1, #0,
        "debug": False,
        "dt":1,
        "order_lag":0,
        "starting_balance":100_000,
        "database_credentials_path": "database/credentials.json",
        "combinatorial_training": True,
        "train_test_split": 0.8,
        "training": True,
        "tickers": None,
    }

    looper = CoreLoop(training_config, env_config)
    looper.run()
