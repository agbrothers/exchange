import numpy as np

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.torch_action_dist import TorchMultiCategorical
from ray.rllib.utils.typing import TensorType, List, Union, Tuple, ModelConfigDict


class TorchMutableMultiCategorical(TorchMultiCategorical):
    def __init__(
        self,
        inputs: List[TensorType],
        model: TorchModelV2,
        input_lens: Union[List[int], np.ndarray, Tuple[int, ...]]=None,
        action_space=None,
    ):
        input_lens = model.action_space.nvec
        super().__init__(inputs, model, input_lens, action_space)

