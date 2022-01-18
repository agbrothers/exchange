from abc import ABC, abstractmethod


class BaseEnv(ABC):

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abstractmethod
    def reset(self, action):
        raise NotImplementedError

    @abstractmethod
    def close(self, action):
        raise NotImplementedError

    @abstractmethod
    def render(self, action):
        raise NotImplementedError

    @abstractmethod
    def seed(self, action):
        raise NotImplementedError

    @abstractmethod
    def action_space(self, action):
        raise NotImplementedError

    @abstractmethod
    def observation_space(self, action):
        raise NotImplementedError
