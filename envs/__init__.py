from gym.envs.registration import register


register(
    id="exchange-v0",
    entry_point="envs.exchange:Exchange",
)
