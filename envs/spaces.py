import numpy as np
from gym.spaces import Box, MultiDiscrete


class MutableBox(Box):
    # Intended for vectorized obs spaces in envs where entity counts 
    # are mutable across steps or episodes

    def __init__(self, low, high, shape=None, dtype=np.float32):
        super().__init__(low, high, shape, dtype)

    def update(self, shape):
        self.__init__(self.low[-1], self.high[-1], shape, self.dtype)


class MutableMultiDiscrete(MultiDiscrete):
    # Intended for vectorized action spaces in envs where action counts
    # that are mutable across steps or episodes

    def __init__(self, nvec, dtype=np.int64):
        super().__init__(nvec, dtype)

    def update(self, nvec):
        self.__init__(nvec, self.dtype)
