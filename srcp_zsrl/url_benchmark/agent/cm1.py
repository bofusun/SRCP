"""
Based on: https://github.com/crowsonkb/k-diffusion
一致性模型作为actor
"""

import torch as th
import torch.nn as nn
import numpy as np
import copy
import math
import typing as tp
import torch.nn.functional as F

class _L2(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        y = math.sqrt(self.dim) * F.normalize(x, dim=1)
        return y


def _nl(name: str, dim: int) -> tp.List[nn.Module]:
    """Returns a non-linearity given name and dimension"""
    if name == "irelu":
        return [nn.ReLU(inplace=True)]
    if name == "relu":
        return [nn.ReLU()]
    if name == "ntanh":
        return [nn.LayerNorm(dim), nn.Tanh()]
    if name == "layernorm":
        return [nn.LayerNorm(dim)]
    if name == "tanh":
        return [nn.Tanh()]
    if name == "L2":
        return [_L2(dim)]
    raise ValueError(f"Unknown non-linearity {name}")


def mlp(*layers: tp.Sequence[tp.Union[int, str]]) -> nn.Sequential:
    """Provides a sequence of linear layers and non-linearities
    providing a sequence of dimension for the neurons, or name of
    the non-linearities
    Eg: mlp(10, 12, "relu", 15) returns:
    Sequential(Linear(10, 12), ReLU(), Linear(12, 15))
    """
    assert len(layers) >= 2
    sequence: tp.List[nn.Module] = []
    assert isinstance(layers[0], int), "First input must provide the dimension"
    prev_dim: int = layers[0]
    for layer in layers[1:]:
        if isinstance(layer, str):
            sequence.extend(_nl(layer, prev_dim))
        else:
            assert isinstance(layer, int)
            sequence.append(nn.Linear(prev_dim, layer))
            prev_dim = layer
    return nn.Sequential(*sequence)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 z_dim, 
                 action_dim,
                 device,
                 feature_dim=50,
                 hidden_dim=1024,
                 t_dim=16,
                 ln=True):

        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            # nn.ReLU(),
            nn.Linear(t_dim * 2, t_dim),
        )

        # input_dim = state_dim + action_dim + t_dim

        # hidden_dim = 1024
        # feature_dim = 512
        
        
        self.obs_net = mlp(state_dim, hidden_dim, "ntanh", feature_dim)
        self.obs_z_net = mlp(state_dim + z_dim, hidden_dim, "ntanh", feature_dim)
        input_dim = 2*feature_dim + action_dim + t_dim

        if ln:

            self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        # nn.LayerNorm(feature_dim),
                                        # nn.Tanh(),
                                        # nn.Mish(),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        # nn.LayerNorm(hidden_dim),
                                        # nn.Mish(),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        # nn.LayerNorm(hidden_dim),
                                        # nn.Mish())
                                        nn.ReLU(inplace=True))
        else:
            self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.Mish(),
                                        #    nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Mish(),
                                        #    nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Mish())

        self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time, state, z):

        t = self.time_mlp(time)
        obs = self.obs_net(state)
        obs_z = self.obs_z_net(th.cat([state, z], dim=-1))

        x = th.cat([x, t, obs, obs_z], dim=-1)
        x = self.mid_layer(x)

        return self.final_layer(x)
    
class MLP6(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 z_dim, 
                 action_dim,
                 device,
                 feature_dim=50,
                 hidden_dim=1024,
                 t_dim=16,
                 ln=True):

        super(MLP6, self).__init__()
        self.device = device
        
        t_dim = z_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            # nn.ReLU(),
            nn.Linear(t_dim * 2, t_dim),
        )

        # input_dim = state_dim + action_dim + t_dim

        # hidden_dim = 1024
        # feature_dim = 512
        
        self.z_net = mlp(z_dim, hidden_dim, "ntanh", z_dim)
        self.z_t_net = mlp(2*z_dim, hidden_dim, "irelu", z_dim)
        
        
        self.obs_net = mlp(state_dim, hidden_dim, "ntanh", feature_dim)
        self.obs_z_net = mlp(state_dim + z_dim, hidden_dim, "ntanh", feature_dim)
        input_dim = 2*feature_dim + action_dim + t_dim

        if ln:

            self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        # nn.LayerNorm(feature_dim),
                                        # nn.Tanh(),
                                        # nn.Mish(),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        # nn.LayerNorm(hidden_dim),
                                        # nn.Mish(),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        # nn.LayerNorm(hidden_dim),
                                        # nn.Mish())
                                        nn.ReLU(inplace=True))
        else:
            self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.Mish(),
                                        #    nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Mish(),
                                        #    nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Mish())

        self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time, state, z):

        t = self.time_mlp(time)
        z = self.z_net(z)
        y = self.z_t_net(th.cat([t, z], dim=-1))
        obs = self.obs_net(state)
        obs_z = self.obs_z_net(th.cat([state, z], dim=-1))

        x = th.cat([x, y, obs, obs_z], dim=-1)
        x = self.mid_layer(x)

        return self.final_layer(x)


class MLP1(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 z_dim, 
                 action_dim,
                 device,
                 feature_dim=50,
                 hidden_dim=1024,
                 t_dim=50,
                 ln=True):

        super(MLP1, self).__init__()
        self.device = device
        
        t_dim = z_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        
        self.z_net = mlp(z_dim, hidden_dim, "ntanh", t_dim)

        # input_dim = state_dim + action_dim + t_dim

        # hidden_dim = 1024
        # feature_dim = 512
        
        
        self.obs_net = mlp(state_dim, hidden_dim, "ntanh", feature_dim)
        input_dim = feature_dim + action_dim + t_dim

        if ln:
            self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        # nn.LayerNorm(feature_dim),
                                        # nn.Tanh(),
                                        # nn.Mish(),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        # nn.LayerNorm(hidden_dim),
                                        # nn.Mish(),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        # nn.LayerNorm(hidden_dim),
                                        # nn.Mish())
                                        nn.ReLU(inplace=True))
        else:
            self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.Mish(),
                                        #    nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Mish(),
                                        #    nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Mish())

        self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time, state, z):

        t = self.time_mlp(time)
        z = self.z_net(z)
        y = z + t
        obs = self.obs_net(state)

        x = th.cat([x, y, obs], dim=-1)
        x = self.mid_layer(x)

        return self.final_layer(x)

class MOE_MLP1(nn.Module):
    """
    MLP Model with Mixture of Experts (MOE) structure
    """
    def __init__(self,
                 state_dim,
                 z_dim, 
                 action_dim,
                 device,
                 feature_dim=50,
                 hidden_dim=1024,
                 t_dim=50,
                 ln=True,
                 n_experts=6,
                 topk=3):
        
        super(MOE_MLP1, self).__init__()
        self.device = device
        self.n_experts = n_experts
        self.topk = topk
        
        t_dim = z_dim

        # Time embedding network (unchanged)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        
        # z network (unchanged)
        self.z_net = mlp(z_dim, hidden_dim, "ntanh", t_dim)
        
        # Observation network (unchanged)
        self.obs_net = mlp(state_dim, hidden_dim, "ntanh", feature_dim)

        # MOE components for processing the concatenated input
        self.router = nn.Sequential(
            nn.Linear(feature_dim + 2*t_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_experts))
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim + action_dim + t_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, action_dim)
            ) for _ in range(n_experts)
        ])
        
        # Noise distribution for router exploration
        self.noise_distr = th.distributions.Normal(
            loc=th.tensor([0.0]*n_experts, device=device), 
            scale=th.tensor([1.0/n_experts]*n_experts, device=device)
        )

        # Final layers (unchanged)
        # self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time, state, z, router_noise=False):
        # Time and z processing (unchanged)
        t = self.time_mlp(time)
        z = self.z_net(z)
        y = z + t
        
        # Observation processing (unchanged)
        obs = self.obs_net(state)
        
        # Concatenate inputs
        x = th.cat([x, y, obs], dim=-1)
        
        # MOE processing
        router_logits = self.router(th.cat([z, t, obs], dim=-1))
        
        if router_noise:
            # Add noise for exploration
            router_logits = router_logits + self.noise_distr.sample()
        
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k sparse gating
        topk_probs, topk_indices = th.topk(router_probs, self.topk, dim=-1)
        sparse_probs = th.zeros_like(router_probs).scatter(
            dim=-1, index=topk_indices, src=topk_probs
        )
        
        # Expert processing
        expert_outputs = th.stack([expert(x) for expert in self.experts], dim=1)
        x = th.sum(expert_outputs * sparse_probs.unsqueeze(-1), dim=1)
        
        return x
        
        
        
def get_generator(generator, num_samples=0, seed=0):
    if generator == "dummy":
        return DummyGenerator()
    else:
        raise NotImplementedError
    
class MLP2(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 z_dim, 
                 action_dim,
                 device,
                 feature_dim=50,
                 hidden_dim=1024,
                 t_dim=50,
                 ln=True):

        super(MLP2, self).__init__()
        self.device = device
        
        t_dim = z_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        
        self.z_net = mlp(z_dim, hidden_dim, "ntanh", t_dim)

        # input_dim = state_dim + action_dim + t_dim

        # hidden_dim = 1024
        # feature_dim = 512
        
        
        self.obs_net = mlp(state_dim, hidden_dim, "ntanh", feature_dim)
        input_dim = feature_dim + action_dim + 2*t_dim

        if ln:
            self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        # nn.LayerNorm(feature_dim),
                                        # nn.Tanh(),
                                        # nn.Mish(),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        # nn.LayerNorm(hidden_dim),
                                        # nn.Mish(),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        # nn.LayerNorm(hidden_dim),
                                        # nn.Mish())
                                        nn.ReLU(inplace=True))
        else:
            self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.Mish(),
                                        #    nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Mish(),
                                        #    nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Mish())

        self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time, state, z):

        t = self.time_mlp(time)
        z = self.z_net(z)
        y = th.cat([z, t], dim=-1)
        obs = self.obs_net(state)

        x = th.cat([x, y, obs], dim=-1)
        x = self.mid_layer(x)

        return self.final_layer(x)

class MLP3(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 z_dim, 
                 action_dim,
                 device,
                 feature_dim=50,
                 hidden_dim=1024,
                 t_dim=50,
                 ln=True):

        super(MLP3, self).__init__()
        self.device = device
        
        t_dim = z_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        
        self.z_net = mlp(z_dim, hidden_dim, "ntanh", z_dim)
        self.z_t_net = mlp(2*z_dim, hidden_dim, "irelu", z_dim)
        
        
        self.obs_net = mlp(state_dim, hidden_dim, "ntanh", feature_dim)
        input_dim = feature_dim + action_dim + t_dim

        if ln:
            self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        # nn.LayerNorm(feature_dim),
                                        # nn.Tanh(),
                                        # nn.Mish(),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        # nn.LayerNorm(hidden_dim),
                                        # nn.Mish(),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        # nn.LayerNorm(hidden_dim),
                                        # nn.Mish())
                                        nn.ReLU(inplace=True))
        else:
            self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.Mish(),
                                        #    nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Mish(),
                                        #    nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Mish())

        self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time, state, z):

        t = self.time_mlp(time)
        z = self.z_net(z)
        y = self.z_t_net(th.cat([t, z], dim=-1))
        obs = self.obs_net(state)

        x = th.cat([x, y, obs], dim=-1)
        x = self.mid_layer(x)

        return self.final_layer(x)

class MOE_MLP3(nn.Module):
    """
    MLP Model with Mixture of Experts (MOE) structure
    """
    def __init__(self,
                 state_dim,
                 z_dim, 
                 action_dim,
                 device,
                 feature_dim=50,
                 hidden_dim=1024,
                 t_dim=50,
                 ln=True,
                 n_experts=6,
                 topk=3):
        
        super(MOE_MLP3, self).__init__()
        self.device = device
        self.n_experts = n_experts
        self.topk = topk
        
        t_dim = z_dim

        # Time embedding network (unchanged)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        
        self.z_net = mlp(z_dim, hidden_dim, "ntanh", z_dim)
        self.z_t_net = mlp(2*z_dim, hidden_dim, "irelu", z_dim)
        
        
        self.obs_net = mlp(state_dim, hidden_dim, "ntanh", feature_dim)
        input_dim = feature_dim + action_dim + t_dim
        
        # MOE components for processing the concatenated input
        self.router = nn.Sequential(
            nn.Linear(feature_dim + t_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_experts))
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim + action_dim + t_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, action_dim)
            ) for _ in range(n_experts)
        ])
        
        # Noise distribution for router exploration
        self.noise_distr = th.distributions.Normal(
            loc=th.tensor([0.0]*n_experts, device=device), 
            scale=th.tensor([1.0/n_experts]*n_experts, device=device)
        )

        # Final layers (unchanged)
        # self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time, state, z, router_noise=False):
        # Time and z processing (unchanged)       
        t = self.time_mlp(time)
        z = self.z_net(z)
        y = self.z_t_net(th.cat([t, z], dim=-1))
        obs = self.obs_net(state)
        
        # Concatenate inputs
        x = th.cat([x, y, obs], dim=-1)
        
        # MOE processing
        router_logits = self.router(th.cat([y, obs], dim=-1))
        
        if router_noise:
            # Add noise for exploration
            router_logits = router_logits + self.noise_distr.sample()
        
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k sparse gating
        topk_probs, topk_indices = th.topk(router_probs, self.topk, dim=-1)
        sparse_probs = th.zeros_like(router_probs).scatter(
            dim=-1, index=topk_indices, src=topk_probs
        )
        
        # Expert processing
        expert_outputs = th.stack([expert(x) for expert in self.experts], dim=1)
        x = th.sum(expert_outputs * sparse_probs.unsqueeze(-1), dim=1)
        
        return x
    
# 修改后的MLP，包含obs_z_net
class MLP4(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 z_dim, 
                 action_dim,
                 device,
                 feature_dim=50,
                 hidden_dim=1024,
                 t_dim=16,
                 ln=True):

        super(MLP4, self).__init__()
        self.device = device
        
        t_dim = z_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        self.obs_net = mlp(state_dim, hidden_dim, "ntanh", feature_dim)
        self.obs_z_net = mlp(state_dim + z_dim, hidden_dim, "ntanh", feature_dim)
        input_dim = 2*feature_dim + action_dim + t_dim

        if ln:

            self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(inplace=True))
        else:
            self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.Mish(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Mish(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Mish())

        self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time, state, z):

        t = self.time_mlp(time)
        obs = self.obs_net(state)
        obs_z = self.obs_z_net(th.cat([state, z], dim=-1))

        x = th.cat([x, t, obs, obs_z], dim=-1)
        x = self.mid_layer(x)

        return self.final_layer(x)

# 修改后的MLP，包含obs_z_net和y=z+t
class MLP5(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 z_dim, 
                 action_dim,
                 device,
                 feature_dim=50,
                 hidden_dim=1024,
                 t_dim=16,
                 ln=True):

        super(MLP5, self).__init__()
        self.device = device
        
        t_dim = z_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.z_net = mlp(z_dim, hidden_dim, "ntanh", z_dim)

        self.obs_net = mlp(state_dim, hidden_dim, "ntanh", feature_dim)
        self.obs_z_net = mlp(state_dim + z_dim, hidden_dim, "ntanh", feature_dim)
        input_dim = 2*feature_dim + action_dim + t_dim

        if ln:

            self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(inplace=True))
        else:
            self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.Mish(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Mish(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Mish())

        self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time, state, z):

        t = self.time_mlp(time)
        repre_z = self.z_net(z)
        y = repre_z + t
        obs = self.obs_net(state)
        obs_z = self.obs_z_net(th.cat([state, z], dim=-1))

        x = th.cat([x, y, obs, obs_z], dim=-1)
        x = self.mid_layer(x)

        return self.final_layer(x)
    
# 修改后的MLP，包含obs_z_net和y=f(z_net(z),t_net(t))
class MLP6(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 z_dim, 
                 action_dim,
                 device,
                 feature_dim=50,
                 hidden_dim=1024,
                 t_dim=16,
                 ln=True):

        super(MLP6, self).__init__()
        self.device = device
        
        t_dim = z_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        
        self.z_net = mlp(z_dim, hidden_dim, "ntanh", z_dim)
        self.z_t_net = mlp(2*z_dim, hidden_dim, "irelu", z_dim)
        

        self.obs_net = mlp(state_dim, hidden_dim, "ntanh", feature_dim)
        self.obs_z_net = mlp(state_dim + z_dim, hidden_dim, "ntanh", feature_dim)
        input_dim = 2*feature_dim + action_dim + t_dim

        if ln:

            self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(inplace=True))
        else:
            self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.Mish(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Mish(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Mish())

        self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time, state, z):

        t = self.time_mlp(time)
        z = self.z_net(z)
        y = self.z_t_net(th.cat([t, z], dim=-1))
        obs = self.obs_net(state)
        obs_z = self.obs_z_net(th.cat([state, z], dim=-1))

        x = th.cat([x, y, obs, obs_z], dim=-1)
        x = self.mid_layer(x)

        return self.final_layer(x)


class FiLM_MLP6(nn.Module):
    def __init__(self,
                 state_dim,
                 z_dim, 
                 action_dim,
                 device,
                 feature_dim=50,
                 hidden_dim=1024,
                 t_dim=16,
                 ln=True):
        super(FiLM_MLP6, self).__init__()
        self.device = device
        
        # 时间编码网络（保持不变）
        t_dim = z_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        
        # 条件编码网络（生成 FiLM 参数）
        self.cond_net = nn.Sequential(
            nn.Linear(state_dim + z_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 2 * hidden_dim)  # 输出 γ 和 β
        )

        # 主网络分支（去除了 obs_z_net，改为 FiLM 调制）
        self.obs_net = mlp(state_dim, hidden_dim, "ntanh", feature_dim)
        input_dim = feature_dim + action_dim + t_dim

        # 中间层（接受 FiLM 调制）
        if ln:
            self.mid_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.mid_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish()
            )

        self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time, state, z):
        # 时间编码
        t = self.time_mlp(time)
        
        # 条件编码（state 和 z 拼接后生成 γ 和 β）
        cond = th.cat([state, z], dim=-1)
        gamma_beta = self.cond_net(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # 分割为 γ 和 β
        
        # 观测编码
        obs = self.obs_net(state)
        
        # 主网络输入
        x = th.cat([x, t, obs], dim=-1)
        x = self.mid_layer(x)
        
        # FiLM 调制
        x = gamma * x + beta  # 特征空间动态缩放和平移
        
        return self.final_layer(x)
    
# ========== 工具模块 ==========
class AdaGNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, t_dim, cond_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.GroupNorm(4, out_dim)
        
        # 门控权重生成
        self.gate = nn.Sequential(
            nn.Linear(t_dim + cond_dim, out_dim * 2),
            nn.Sigmoid()  # 输出[0,1]范围
        )
        
        # 联合特征投影
        self.joint_proj = nn.Sequential(
            nn.Linear(t_dim + cond_dim, out_dim * 2),
            nn.GELU()
        )

    def forward(self, x, timestep, condition):
        x = self.linear(x)
        x = self.norm(x)
        
        # 拼接并计算
        joint = th.cat([timestep, condition], dim=-1)
        gate = self.gate(joint)  # 动态权重
        scale, shift = self.joint_proj(joint).chunk(2, dim=-1)
        
        # 门控融合
        scale = scale * gate[:, :scale.size(1)]
        shift = shift * gate[:, shift.size(1):]
        
        return x * (1 + scale) + shift
    
# 修改后的MLP，包含obs_z_net和y=f(z_net(z),t_net(t))
class MLP7(nn.Module):
    def __init__(self,
                state_dim,
                z_dim, 
                action_dim,
                device,
                feature_dim=128,
                hidden_dim=1024,
                t_dim=64,
                num_heads=4,
                ln=True):
        
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim

        # 1. Encoders
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.Mish(),
            nn.Linear(t_dim * 4, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim)
        )
        self.z_net = mlp(z_dim, hidden_dim, "ntanh", 2*feature_dim, "irelu")
        self.obs_net = mlp(state_dim, hidden_dim, "ntanh", 2*feature_dim, "irelu")

        # 2. Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 3. Backbone
        self.input_proj = nn.Linear(action_dim + feature_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            AdaGNBlock(hidden_dim, hidden_dim, t_dim, feature_dim),
            AdaGNBlock(hidden_dim, hidden_dim//2, t_dim, feature_dim),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            AdaGNBlock(hidden_dim//2, hidden_dim//4, t_dim, feature_dim)
        ])
        self.final_layer = mlp(hidden_dim//4, hidden_dim//8, "irelu", action_dim)

    def forward(self, x, time, state, z):
        # Encoding
        t = self.time_mlp(time)
        z_enc = self.z_net(z)
        obs_enc = self.obs_net(state)

        # Cross-Attention
        z_key = z_enc[..., :self.feature_dim].unsqueeze(1)
        z_val = z_enc[..., self.feature_dim:].unsqueeze(1)
        obs_query = obs_enc[..., :self.feature_dim].unsqueeze(1)
        fused, _ = self.cross_attn(obs_query, z_key, z_val)
        fused = fused.squeeze(1)  # [B, feature_dim]

        # Processing
        x = th.cat([x, fused], dim=-1)
        x = self.input_proj(x)
        
        # Backbone with residuals
        for block in self.blocks:
            residual = x
            if isinstance(block, AdaGNBlock):
                x = block(x, t, fused)  # 仅用fused作为条件
            else:
                x = block(x)
            if x.shape == residual.shape:
                x = x + residual
            x = F.gelu(x)
            
        return self.final_layer(x)

class MLP8(nn.Module):
    def __init__(self, state_dim, z_dim, action_dim, device, 
                feature_dim=128, hidden_dim=1024, t_dim=32, ln=True):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim  # 新增属性
        
        # 1. 时间编码（增强版）-> 输出维度扩大以适应调制
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.Mish(),
            nn.Linear(t_dim * 4, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, feature_dim)  # 输出足够维度用于调制
        )
        
        # 2. 条件编码（与时间步解耦）
        self.z_net = mlp(z_dim, hidden_dim, "ntanh", feature_dim * 2)
        self.obs_net = mlp(state_dim, hidden_dim, "ntanh", feature_dim * 2)
        
        # 3. 交叉注意力融合（仅处理技能和状态）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            batch_first=True
        )
        
        # 4. 主干网络（移除时间步输入）
        input_dim = feature_dim + action_dim  # 不再直接拼接t
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish()
        )
        
        # 5. 时间步调制器（FiLM风格）
        self.time_scale = nn.Linear(feature_dim, hidden_dim)
        self.time_shift = nn.Linear(feature_dim, hidden_dim)
        
        # 6. 输出层
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, x, time, state, z):
        # === 阶段1：独立编码 ===
        # 时间编码（独立路径）
        t = self.time_mlp(time)  # [B, hidden_dim*2]
        
        # 条件编码（技能+状态）
        z_enc = self.z_net(z)  # [B, feature_dim*2]
        obs_enc = self.obs_net(state)  # [B, feature_dim*2]
        
        # === 阶段2：条件融合（不含时间步）===
        z_key_val = z_enc.view(-1, 2, self.feature_dim)  # [B, 2, D]
        obs_query = obs_enc.view(-1, 2, self.feature_dim)
        fused, _ = self.cross_attn(obs_query, z_key_val, z_key_val)
        fused = fused.mean(dim=1)  # [B, D]
        
        # === 阶段3：主干处理（时间步作为调制信号）===
        # 合并动作和条件（不包含t）
        x = th.cat([x, fused], dim=-1)
        x = self.mid_layer(x)  # [B, hidden_dim]
        
        # 时间步全局调制（FiLM）
        scale = self.time_scale(t)  # [B, hidden_dim]
        shift = self.time_shift(t)
        x = x * (1 + scale) + shift  # 调制
        
        # === 阶段4：输出 ===
        return self.final_layer(x)



class MLP9(nn.Module):
    def __init__(self, state_dim, z_dim, action_dim, device, 
                feature_dim=128, hidden_dim=1024, t_dim=32):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim

        # 1. 时间编码（输出 feature_dim）
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.Mish(),
            nn.Linear(t_dim * 4, feature_dim),  # 输出 feature_dim
            nn.LayerNorm(feature_dim)
        )
        
        # 2. 自适应参数生成（feature_dim -> 6*feature_dim）
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feature_dim, 6 * feature_dim, bias=True)
        )

        # 3. 特征编码网络
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU()
        )
        self.obs_z_net = nn.Sequential(
            nn.Linear(state_dim + z_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU()
        )
        self.norm1 = nn.LayerNorm(feature_dim)  # 用于query
        self.norm2 = nn.LayerNorm(feature_dim)  # 用于key/value
        self.norm3 = nn.LayerNorm(feature_dim)  # 用于MLP输入

        # 4. 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )

        # 5. 主干网络
        self.mid_layer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        # 6. 输出层
        self.final_layer = nn.Sequential(
            nn.Linear(feature_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, x, time, state, z):
        # === 阶段1：参数生成 ===
        t = self.time_mlp(time)  # [B, feature_dim]
        params = self.adaLN_modulation(t)  # [B, 6*feature_dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=1)

        # === 阶段2：特征编码与MSA调制 ===
        x = self.action_net(x)  # [B, feature_dim]
        obs_z = self.obs_z_net(th.cat([state, z], dim=-1))  # [B, feature_dim]
        
        # 调制query特征
        modulated_x = self.norm1(x) * (1 + scale_msa) + shift_msa
        obs_z = self.norm2(obs_z)
        
        # 注意力计算
        attn_out, _ = self.cross_attn(
            query=modulated_x.unsqueeze(1),
            key=obs_z.unsqueeze(1),
            value=obs_z.unsqueeze(1)
        )
        attn_out = gate_msa.unsqueeze(1) * attn_out  # 门控限制
        
        # 残差融合
        fused = x + attn_out.squeeze(1) 

        # === 阶段3：MLP处理 ===
        modulated_fused = self.norm3(fused) * (1 + scale_mlp) + shift_mlp
        mid_out = self.mid_layer(modulated_fused)
        
        # 门控残差
        output = fused + gate_mlp * mid_out
        return self.final_layer(output)



class ValueMLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 z_dim, 
                 action_dim,
                 device,
                 feature_dim=50,
                 hidden_dim=1024,
                 t_dim=16,
                 ln=True):
        

        super(ValueMLP, self).__init__()
        self.device = device
        
        t_dim = z_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            # nn.ReLU(),
            nn.Linear(t_dim * 2, t_dim),
        )

        # input_dim = state_dim + action_dim + t_dim

        # hidden_dim = 1024
        # feature_dim = 512
        
        self.z_net = mlp(z_dim, hidden_dim, "ntanh", t_dim)

        # input_dim = state_dim + action_dim + t_dim

        # hidden_dim = 1024
        # feature_dim = 512
        
        
        self.obs_action_net = mlp(state_dim + action_dim, hidden_dim, "ntanh", feature_dim, "irelu")
        input_dim = feature_dim + z_dim + t_dim

        seq = [input_dim, hidden_dim, "irelu", hidden_dim,  "irelu", z_dim]
        self.F1 = mlp(*seq)
        self.F2 = mlp(*seq)

    def forward(self, x, time, state, z, action):

        t = self.time_mlp(time)
        z = self.z_net(z)
        y = z + t
        obs_action = self.obs_action_net(th.cat([state, action], dim=-1))

        x = th.cat([x, y, obs_action], dim=-1)
        F1 = self.F1(x)
        F2 = self.F2(x)

        return F1, F2
    
def get_generator(generator, num_samples=0, seed=0):
    if generator == "dummy":
        return DummyGenerator()
    else:
        raise NotImplementedError

class DummyGenerator:
    def randn(self, *args, **kwargs):
        return th.randn(*args, **kwargs)

    def randint(self, *args, **kwargs):
        return th.randint(*args, **kwargs)

    def randn_like(self, *args, **kwargs):
        return th.randn_like(*args, **kwargs)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def append_zero(x):
    return th.cat([x, x.new_zeros([1])])

def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = th.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = th.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings

class ConsistencyModel(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(ConsistencyModel, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
class ConsistencyModel6(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(ConsistencyModel6, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device
        
        th.manual_seed(42)

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP6(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, skill, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
class Seperate_ConsistencyModel6(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
        guidance_scale=3.0,
    ):
        super(Seperate_ConsistencyModel6, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho
        self.guidance_scale = guidance_scale          # 推理时引导强度

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP6(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised
        
    def denoise_infer(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        zero_skill = th.zeros_like(skill)
        if return_dict:
            cond_model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
            uncond_model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, zero_skill, return_dict)
        else:
            cond_model_output = model(c_in * x_t, rescaled_t, state, skill)
            uncond_model_output = model(c_in * x_t, rescaled_t, state, zero_skill)
        model_output = uncond_model_output + self.guidance_scale * (cond_model_output - uncond_model_output)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise_infer(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise_infer(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise_infer(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise_infer(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, skill, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
class Seperate_ConsistencyModel6half(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
        guidance_scale = 0.5,
    ):
        super(Seperate_ConsistencyModel6half, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho
        self.guidance_scale = guidance_scale          # 推理时引导强度

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP6(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised
        
    def denoise_infer(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        zero_skill = th.zeros_like(skill)
        if return_dict:
            cond_model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
            uncond_model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, zero_skill, return_dict)
        else:
            cond_model_output = model(c_in * x_t, rescaled_t, state, skill)
            uncond_model_output = model(c_in * x_t, rescaled_t, state, zero_skill)
        model_output = uncond_model_output + self.guidance_scale * (cond_model_output - uncond_model_output)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise_infer(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise_infer(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise_infer(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise_infer(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0

class Seperate_ConsistencyModel6_new(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
        guidance_scale=2.0,
    ):
        super(Seperate_ConsistencyModel6_new, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho
        self.guidance_scale = guidance_scale          # 推理时引导强度

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP6(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised
        
    def denoise_infer(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        zero_skill = th.zeros_like(skill)
        if return_dict:
            cond_model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
            uncond_model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, zero_skill, return_dict)
        else:
            cond_model_output = model(c_in * x_t, rescaled_t, state, skill)
            uncond_model_output = model(c_in * x_t, rescaled_t, state, zero_skill)
        model_output = uncond_model_output + self.guidance_scale * (cond_model_output - uncond_model_output)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise_infer(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise_infer(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise_infer(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise_infer(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
class Seperate_ConsistencyModel6_3(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=False,
        ln=False,
        guidance_scale=3.0,
    ):
        super(Seperate_ConsistencyModel6_3, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho
        self.guidance_scale = guidance_scale          # 推理时引导强度

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP6(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised
        
    def denoise_infer(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        zero_skill = th.zeros_like(skill)
        if return_dict:
            cond_model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
            uncond_model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, zero_skill, return_dict)
        else:
            cond_model_output = model(c_in * x_t, rescaled_t, state, skill)
            uncond_model_output = model(c_in * x_t, rescaled_t, state, zero_skill)
        model_output = uncond_model_output + self.guidance_scale * (cond_model_output - uncond_model_output)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise_infer(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise_infer(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise_infer(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise_infer(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
class Energy_ConsistencyModel6(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
        guidance_scale=3.0,
    ):
        super(Energy_ConsistencyModel6, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho
        self.guidance_scale = guidance_scale          # 推理时引导强度

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP6(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised
        
    def denoise_infer(self, successor_net, model, x_t, sigmas, state, skill, return_dict=False):
        x_t.requires_grad_(True) 
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)

        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        F1, F2 = successor_net(state, skill, denoised)
        Q1 = th.einsum('sd, sd -> s', F1, skill)
        Q2 = th.einsum('sd, sd -> s', F2, skill)
        Q = th.min(Q1, Q2)
        lmbda = 1/Q.abs().mean().detach()
        energy = -lmbda*Q  # shape: [batch_size]
        energy_grad = th.autograd.grad(energy.sum(), denoised, create_graph=True)[0]
        denoised = denoised - alpha * energy_grad
        
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, successor_net, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise_infer(successor_net, self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise_infer(successor_net, self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, successor_net, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise_infer(successor_net, self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise_infer(successor_net, self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, successor_net, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(successor_net, state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(successor_net, state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
class Seperate_ConsistencyModel6_1(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
        guidance_scale=5.0,
    ):
        super(Seperate_ConsistencyModel6_1, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho
        self.guidance_scale = guidance_scale          # 推理时引导强度

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP6(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised
        
    def denoise_infer(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        zero_skill = th.zeros_like(skill)
        if return_dict:
            cond_model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
            uncond_model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, zero_skill, return_dict)
        else:
            cond_model_output = model(c_in * x_t, rescaled_t, state, skill)
            uncond_model_output = model(c_in * x_t, rescaled_t, state, zero_skill)
        model_output = uncond_model_output + self.guidance_scale * (cond_model_output - uncond_model_output)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise_infer(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise_infer(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise_infer(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise_infer(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
class Seperate_ConsistencyModel6_2(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
        guidance_scale=5.0,
    ):
        super(Seperate_ConsistencyModel6_2, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho
        self.guidance_scale = guidance_scale          # 推理时引导强度

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP6(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised
        
    def denoise_infer(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        zero_skill = th.zeros_like(skill)
        if return_dict:
            cond_model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
            uncond_model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, zero_skill, return_dict)
        else:
            cond_model_output = model(c_in * x_t, rescaled_t, state, skill)
            uncond_model_output = model(c_in * x_t, rescaled_t, state, zero_skill)
        model_output = uncond_model_output + self.guidance_scale * (cond_model_output - uncond_model_output)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise_infer(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise_infer(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise_infer(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise_infer(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
        
class FiLM_ConsistencyModel6(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(FiLM_ConsistencyModel6, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = FiLM_MLP6(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
        
class ConsistencyModel7(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(ConsistencyModel7, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP7(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, num_heads=4, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
class ConsistencyModel8(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(ConsistencyModel8, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP8(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
class ConsistencyModel9(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(ConsistencyModel9, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP9(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
# 修改后的一致性模型
class ConsistencyModel1(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(ConsistencyModel1, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP4(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
# 修改后的一致性模型+解耦
class ConsistencyModel1_1(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(ConsistencyModel1_1, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP4(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        skill = th.zeros_like(skill)
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        skill = th.zeros_like(skill)
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
        
# 修改后的一致性模型+y=z+t
class ConsistencyModel2(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(ConsistencyModel2, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP5(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
# 修改后的一致性模型+解耦
class ConsistencyModel2_1(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(ConsistencyModel2_1, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP5(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        skill = th.zeros_like(skill)
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        skill = th.zeros_like(skill)
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
        
# 修改后的一致性模型+y=f(z,t)
class ConsistencyModel3(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(ConsistencyModel3, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP6(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0




class ConsistencyModel_condition(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(ConsistencyModel_condition, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP1(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
class ConsistencyModel_condition1(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(ConsistencyModel_condition1, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP1(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps
        
        skill = th.zeros_like(skill)

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        
        skill = th.zeros_like(skill)
        
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
class ConsistencyModel_condition_together(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(ConsistencyModel_condition_together, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP3(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
class ConsistencyModel_condition_together1(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(ConsistencyModel_condition_together1, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP3(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps
        
        skill = th.zeros_like(skill)

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        
        skill = th.zeros_like(skill)
        
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
class ConsistencyModel_condition_cat(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(ConsistencyModel_condition_cat, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP2(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0

class MOE_ConsistencyModel_condition(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(MOE_ConsistencyModel_condition, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MOE_MLP1(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
class MOE_ConsistencyModel_condition_together(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(MOE_ConsistencyModel_condition_together, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MOE_MLP3(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None):
            return self.denoise(target_model, x, t, state, skill)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state, skill)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None):
            return self.denoise(self.model, x, t, state, skill)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state, skill)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, skill, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state, skill)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, skill, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill)[1]
    
    def sample_multistep(self, state, skill, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state, skill)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise(self.model, x, t * s_in, state, skill)[1]

        return x
    
    def forward(self, state, skill, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, skill, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, skill, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0
        
class ConsistencyValueModel(nn.Module):
    def __init__(
        self,
        state_dim,
        skill_dim, 
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(ConsistencyValueModel, self).__init__()
        self.action_dim = action_dim
        self.z_dim = skill_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = ValueMLP(state_dim=state_dim, z_dim =skill_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        skill, 
        action, 
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None, action=None):
            model_output1, model_output2, denoised1, denoised2 = self.denoise(self.model, x, t, state, skill, action)
            return denoised1, denoised2

        @th.no_grad()
        def target_denoise_fn(x, t, state=None, skill=None, action=None):
            target_model_output1, target_model_output2, target_denoised1, target_denoised2 = self.denoise(self.model, x, t, state, skill, action)
            return target_denoised1, target_denoised2

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller1, distiller2 = denoise_fn(x_t, t, state, skill, action)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target1, distiller_target2 = target_denoise_fn(x_t2, t2, state, skill, action)
        distiller_target1, distiller_target2 = distiller_target1.detach(), distiller_target2.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller1 - distiller_target1) ** 2 + (distiller2 - distiller_target2) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, skill, action, noise=None, td_weights=None):
        num_scales = self.steps
        # state = th.cat([state, skill], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None, skill=None, action=None):
            model_output1, model_output2, denoised1, denoised2 = self.denoise(self.model, x, t, state, skill, action)
            return denoised1, denoised2

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller1, distiller2 = denoise_fn(x_t, t, state, skill, action)
        recon_diffs = (distiller1 - x_start) ** 2 + (distiller2 - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, skill, action, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        model_output1, model_output2 = model(c_in * x_t, rescaled_t, state, skill, action)
        denoised1 = c_out * model_output1 + c_skip * x_t
        denoised2 = c_out * model_output2 + c_skip * x_t
        
        return model_output1, model_output2, denoised1, denoised2

    def sample(self, state, skill, action, eval=False):
        if self.sampler == "onestep":  
            denoised1, denoised2 = self.sample_onestep(state, skill, action, eval=eval)
        elif self.sampler == "multistep":
            denoised1, denoised2 = self.sample_multistep(state, skill, action, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        return denoised1, denoised2
    
    def sample_onestep(self, state, skill, action, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.z_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        model_output1, model_output2, denoised1, denoised2 = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, action)
        return denoised1, denoised2
    
    def sample_multistep(self, state, skill, action, eval=False):
        x_T = self.generator.randn((state.shape[0], self.z_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state, skill, action)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        model_output1, model_output2, denoised1, denoised2 = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, skill, action)

        return denoised1, denoised2
    
    def forward(self, state, skill, action, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        # state = th.cat([state, skill], dim=-1)
        if multistep:
            x_0, x_1 = self.sample_multistep(state, skill, action, eval=eval)
        else:
            x_0, x_1 = self.sample_onestep(state, skill, action, eval=eval)

        return x_0, x_1