import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from envs.exchange import Exchange


env_config = {
    "seed":0,
    "dt":1,
    "order_lag":0,
    "starting_balance":100_000,
    "tickers":[
        "GOOG",
        "MSFT",
        "AAPL",
    ],
}


# Parallel environments
env = make_vec_env("exchange-v0", n_envs=1, env_kwargs={"env_config":env_config})

# model = PPO(
#     "MlpPolicy", 
#     env, 
#     verbose=1,
#     n_steps=389*8
# )
model = PPO.load("ppo_exchange_0")
model.env = env
model.learn(total_timesteps=10_000_000)

model.save("ppo_exchange")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_exchange_0")
env = gym.make(id="exchange-v0", env_config=env_config)

# Have this go through each day with cumulative balance as an eval round
for i in range(10):
    obs = env.reset(eval=True)
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done: 
            break
        # env.render()


