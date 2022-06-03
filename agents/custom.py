import math
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1 / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions):
        sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]


class GRU(torch.nn.Module):
    def __init__(self, hidden_size, b_g=0.1):
        super().__init__()
        self.W_r = torch.nn.Linear(hidden_size, hidden_size)
        self.U_r = torch.nn.Linear(hidden_size, hidden_size)
        self.W_z = torch.nn.Linear(hidden_size, hidden_size)
        self.U_z = torch.nn.Linear(hidden_size, hidden_size)
        self.W_g = torch.nn.Linear(hidden_size, hidden_size)
        self.U_g = torch.nn.Linear(hidden_size, hidden_size)
        self.b_g = b_g
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, y):
        r = self.sigmoid(self.W_r(y) + self.U_r(x))
        z = self.sigmoid(self.W_z(y) + self.U_z(x) - self.b_g)
        h = self.tanh(self.W_g(y) + self.U_g(torch.mul(r, x)))
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g


class GTrXLEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, dim_model, num_heads, dim_feedforward, dropout):
        super().__init__(
            d_model=dim_model, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            layer_norm_eps=1e-6, 
            batch_first=True,
        )
        self.gate_mha = GRU(dim_model)
        self.gate_mlp = GRU(dim_model)

    def forward(self, x, memory, mask=None):  
        batch, seq_len, d_model = x.shape
        x_skip = x

        # Combine input sequence and memory then normalize
        y_hat = self.norm1(torch.cat((x, memory), axis=1))
        # Recover the normalized input sequence
        y = y_hat[:,:seq_len].clone()

        # Multiheaded Attention 
        y = self.self_attn(y, y_hat, y_hat, mask)
        y = self.dropout1(y)
        y = self.activation(y)
        y = self.gate_mha(x_skip, y)

        # Feed Forward Network
        x_skip = y.clone()
        y = self.norm2(y)
        y = self.activation(self.linear1(y))
        y = self.dropout2(self.linear2(y))
        y = self.activation(y)
        y = self.gate_mlp(x_skip, y)
        return y


class GTrXLEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__(encoder_layer, num_layers, norm)
    
    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        output = x
        hidden_state = []

        for i,mod in enumerate(self.layers):
            output = mod(output, memory[i])
            hidden_state.append(output)
        return torch.stack(hidden_state)


class GTrXL(nn.Module):
    '''
    Implementation of GTrXL without relative positional encoding
    '''
    def __init__(
            self,
            dim_model,
            dim_feedforward,
            dim_embedding,
            dim_token,
            dim_vocab,
            num_layers,
            num_heads,
            dropout,
            mem_len=None,
        ):
        nn.Module.__init__(self)
        self.dim_model = dim_model # hidden_size
        self.dim_feedforward = dim_feedforward
        self.dim_embedding = dim_embedding
        self.dim_token = dim_token
        self.dim_vocab = dim_vocab
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # OBSERVATION SPACE LAYERS
        encoder_layer = GTrXLEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.memory = None

    def init_memory(self, x):
        if self.mem_len > 0:
            num_batches = len(x)
            param = next(self.parameters())
            memory = torch.zeros(self.num_layers, num_batches, self.mem_len, self.dim_model, dtype=param.dtype, device=param.device)
            return memory
        else:
            return None

    def update_memory(self, hidden_state, memory):
        # Update memory with new hidden state
        if memory is None: 
            return None
        assert len(hidden_state) == len(memory)

        with torch.no_grad():
            # [layer, batch, seq_len, dim_model]
            # mem_len ~ number of hidden_state tokens in memory
            new_memory = torch.cat((memory, hidden_state), axis=2)
            new_memory = new_memory[:,:,:-self.mem_len,:].detach()
        return new_memory

    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)
        signal = F.pad(signal, (0, 0, 0, self.d_model % 2))
        signal = signal.view(1, max_length, self.d_model)
        return signal

    def forward(self, x):
        if self.memory is None:
            self.memory = self.init_memory(x)

        hidden_state = self.encoder(x, self.memory)
        self.memory = self.update_memory(hidden_state, self.memory)
        output = hidden_state[-1]
        return output


class RLlibWrapper(nn.Module, TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):

        print(" INIT CUSTOM MODEL ")
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.policy_network = GTrXL(
            dim_model = model_config["custom_model_config"]["dim_model"],
            dim_feedforward = model_config["custom_model_config"]["dim_feedforward"],
            dim_embedding = model_config["custom_model_config"]["dim_embedding"],
            dim_token = model_config["custom_model_config"]["dim_token"],
            dim_vocab = model_config["custom_model_config"]["dim_vocab"],
            num_layers = model_config["custom_model_config"]["num_layers"],
            num_heads = model_config["custom_model_config"]["num_heads"],
            dropout = model_config["custom_model_config"]["dropout"],
            mem_len = model_config["custom_model_config"]["mem_len"],
        )
        self.value_network = GTrXL(
            dim_model = model_config["custom_model_config"]["dim_model"],
            dim_feedforward = model_config["custom_model_config"]["dim_feedforward"],
            dim_embedding = model_config["custom_model_config"]["dim_embedding"],
            dim_token = model_config["custom_model_config"]["dim_token"],
            dim_vocab = model_config["custom_model_config"]["dim_vocab"],
            num_layers = model_config["custom_model_config"]["num_layers"],
            num_heads = model_config["custom_model_config"]["num_heads"],
            dropout = model_config["custom_model_config"]["dropout"],
            mem_len = model_config["custom_model_config"]["mem_len"],
        )


    @override(TorchModelV2)
    def forward(self, 
            input_dict, 
            state=[], 
            seq_lens=None
        ):
        # print("Policy Forward")
        x = input_dict["obs"]
        policy_logits = self.policy_network.forward(x)
        value_logits = self.value_network.forward(x)

        # Map policy logits to action space
        # Map value logits to singular value
        # logits = f(policy_logits)
        # self._cur_value = g(value)
        return policy_logits, state


    @override(TorchModelV2)
    def value_function(self) -> torch.tensor:
        # print("Value Function Forward")
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value


if __name__ == "__main__":

    model = GTrXL(
        dim_model=256,
        dim_feedforward=512,
        dim_embedding=256,
        dim_token=10,
        dim_vocab=1024,
        num_layers=1,
        num_heads=4,
        dropout=0.1,
        mem_len=32,
    )

    model.forward()