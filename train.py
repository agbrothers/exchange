import yaml
from copy import copy

import ray
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch
torch, nn = try_import_torch()

from envs.exchange import Exchange
# from agents.space_time_attention_network import SpaceTimeAttentionNetwork
# from agents.wrappers import register
# register()



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




if __name__=="__main__":

    ray.init()

    # Load config
    config = None
    with open("configs/train.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    trainer_config = copy(config["trainer_config"])


    ray.rllib.models.ModelCatalog.register_custom_model(
        "CustomModel", CustomModel)
    trainer_config['model']['custom_model'] = 'CustomModel'

    from envs.action_distribution import TorchMutableMultiCategorical
    ray.rllib.models.ModelCatalog.register_custom_action_dist(
        "TorchMutableMultiCategorical", TorchMutableMultiCategorical)


    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.

    # Create our RLlib Trainer.
    from ray.rllib.agents import ppo
    ppo_config = ppo.DEFAULT_CONFIG
    ppo_config.update(trainer_config)
    trainer = PPOTrainer(config=ppo_config)
    # custom_policy = ppo.PPOTorchPolicy(obs_space, action_space, config=ppo_config)

    # Can optionally call trainer.restore(path) to load a checkpoint.
    for i in range(100):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result))

        if i % 100 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)

    # Evaluate the trained Trainer (and render each timestep to the shell's
    # output).
    trainer.evaluate()


    # Also, in case you have trained a model outside of ray/RLlib and have created
    # an h5-file with weight values in it, e.g.
    # my_keras_model_trained_outside_rllib.save_weights("model.h5")
    # (see: https://keras.io/models/about-keras-models/)

    # ... you can load the h5-weights into your Trainer's Policy's ModelV2
    # (tf or torch) by doing:
    # trainer.import_model("my_weights.h5")
    # NOTE: In order for this to work, your (custom) model needs to implement
    # the `import_from_h5` method.
    # See https://github.com/ray-project/ray/blob/master/rllib/tests/test_model_imports.py
    # for detailed examples for tf- and torch trainers/models.


