from gym.envs.registration import register
from ray.tune.registry import register_env

from envs.exchange import Exchange

# gym registration
register(
    id="exchange-v0",
    entry_point="envs.exchange:Exchange",
)
# rllib registration
register_env('exchange-v0', lambda config: Exchange(config))