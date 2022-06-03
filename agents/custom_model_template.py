from copy import copy

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch
torch, nn = try_import_torch()

from envs.exchange import Exchange


class CustomModel(nn.Module, TorchModelV2):
    def __init__(
            self, obs_space, action_space, num_outputs, model_config, name):

        print(" INIT CUSTOM MODEL ")
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.linear = nn.Linear(29, 2 + 2 + 10)
        self._cur_value = None

    @override(TorchModelV2)
    def forward(self, 
            input_dict, 
            state=[], 
            seq_lens=None
        ):
        x = input_dict["obs"][:,2:]
        x = x.view((x.shape[0], -1, 29))
        x = self.linear(x)
        logits = x.view((x.shape[0], -1))
        self.num_outputs = x.shape
        self._cur_value = torch.zeros(1)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> torch.tensor:
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value