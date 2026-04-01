import math
import typing as tp

import torch
from torch import nn
import torch.nn.functional as F
from url_benchmark import utils


class OnlineCov(nn.Module):
    def __init__(self, mom: float, dim: int) -> None:
        super().__init__()
        self.mom = mom  # momentum
        self.count = torch.nn.Parameter(torch.LongTensor([0]), requires_grad=False)
        self.cov: tp.Any = torch.nn.Parameter(torch.zeros((dim, dim), dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.count += 1  # type: ignore
            self.cov.data *= self.mom
            self.cov.data += (1 - self.mom) * torch.matmul(x.T, x) / x.shape[0]
        count = self.count.item()
        cov = self.cov / (1 - self.mom**count)
        return cov


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


class Actor(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim,
                 preprocess=False, add_trunk=True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.preprocess = preprocess

        if self.preprocess:
            self.obs_net = mlp(self.obs_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            self.obs_z_net = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            if not add_trunk:
                self.trunk: nn.Module = nn.Identity()
                feature_dim = 2 * feature_dim
            else:
                self.trunk = mlp(2 * feature_dim, hidden_dim, "irelu")
                feature_dim = hidden_dim
        else:
            self.trunk = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh",
                             hidden_dim, "irelu",
                             hidden_dim, "irelu")
            feature_dim = hidden_dim

        self.policy = mlp(feature_dim, hidden_dim, "irelu", self.action_dim)
        self.apply(utils.weight_init)
        # initialize the last layer by zero
        # self.policy[-1].weight.data.fill_(0.0)

    def forward(self, obs, z, std):
        assert z.shape[-1] == self.z_dim

        if self.preprocess:
            obs_z = self.obs_z_net(torch.cat([obs, z], dim=-1))
            obs = self.obs_net(obs)
            h = torch.cat([obs, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z], dim=-1)
        if hasattr(self, "trunk"):
            h = self.trunk(h)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

class MOEActor(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim,
                 n_experts=4, topk=2, preprocess=False, add_trunk=True, device='cuda') -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.preprocess = preprocess
        self.n_experts = n_experts
        self.topk = topk
        
        # 噪声分布用于路由器探索
        self.noise_distr = torch.distributions.Normal(
            loc=torch.tensor([0.0]*n_experts, device=device), 
            scale=torch.tensor([1.0/n_experts]*n_experts, device=device)
        )

        # 预处理网络保持不变
        if self.preprocess:
            self.obs_net = mlp(self.obs_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            self.obs_z_net = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            if not add_trunk:
                self.trunk: nn.Module = nn.Identity()
                trunk_output_dim = 2 * feature_dim
            else:
                self.trunk = mlp(2 * feature_dim, hidden_dim, "irelu")
                trunk_output_dim = hidden_dim
        else:
            self.trunk = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh",
                           hidden_dim, "irelu",
                           hidden_dim, "irelu")
            trunk_output_dim = hidden_dim

        # 用MOE替换原来的policy网络
        # 路由器网络
        # self.router = nn.Linear(trunk_output_dim, n_experts)
        self.router = nn.Sequential(nn.Linear(trunk_output_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, n_experts))
        
        # 专家网络 - 每个专家都有自己的均值预测网络
        self.mean_experts = nn.ModuleList([
            mlp(trunk_output_dim, hidden_dim, "irelu", self.action_dim) 
            for _ in range(n_experts)
        ])
        
        self.apply(utils.weight_init)
        # 初始化路由器权重
        nn.init.zeros_(self.router.weight)
        # 初始化最后一个专家层权重为零
        for expert in self.mean_experts:
            expert[-1].weight.data.fill_(0.0)

    def forward(self, obs, z, std, router_noise=False):
        assert z.shape[-1] == self.z_dim

        # 预处理阶段保持不变
        if self.preprocess:
            obs_z = self.obs_z_net(torch.cat([obs, z], dim=-1))
            obs = self.obs_net(obs)
            h = torch.cat([obs, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z], dim=-1)
        
        if hasattr(self, "trunk"):
            h = self.trunk(h)

        # MOE路由逻辑
        router_logits = self.router(h)

        if router_noise:
            # 添加噪声促进探索
            noisy_logits = router_logits + self.noise_distr.sample()
            
            # 计算路由器重要性指标
            importance = F.softmax(noisy_logits, dim=-1).sum(0)
            self.router_importance = (torch.std(importance)/torch.mean(importance))**2
            
            # 计算路由器负载指标
            threshold = torch.max(noisy_logits, dim=-1).values
            load = (1 - self.noise_distr.cdf(threshold.unsqueeze(1) - router_logits)).sum(0)
            self.router_load = (torch.std(load)/torch.mean(load))**2
            
            router_logits = noisy_logits
            self.z_loss = (1 / h.shape[0]) * torch.square(torch.exp(router_logits).sum(1)).sum(0)

        # 获取topk专家
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, self.topk, dim=-1)
        sparse_probs = torch.zeros_like(router_probs).scatter_(
            index=topk_indices, src=topk_probs, dim=-1)
        
        # 记录专家使用情况
        # if h.shape[0] == 1:  # 如果是单个样本
        #     self.episodic_expert_count[topk_indices] += 1

        # 并行计算所有专家输出
        expert_outputs = torch.stack([expert(h) for expert in self.mean_experts], dim=1)
        
        # 加权组合专家输出
        mu = torch.sum(expert_outputs * sparse_probs.unsqueeze(-1), dim=1)
        mu = torch.tanh(mu)  # 保持在[-1,1]范围内
        
        # 使用固定的标准差
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

class MOEActor1(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim,
                 n_experts=6, topk=2, preprocess=False, add_trunk=True, device='cuda') -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.preprocess = preprocess
        self.n_experts = n_experts
        self.topk = topk

        # MOE components for obs_z processing
        # self.router = nn.Linear(obs_dim + z_dim, n_experts)  # 路由网络
        self.router = nn.Sequential(nn.Linear(obs_dim + z_dim, int(hidden_dim/2)),
                                    nn.ReLU(),
                                    nn.Linear(int(hidden_dim/2), n_experts))
        self.experts = nn.ModuleList([
            mlp(obs_dim + z_dim, int(hidden_dim/2), "ntanh", feature_dim, "irelu")
            for _ in range(n_experts)
        ])
        
        # 噪声分布用于路由器探索
        self.noise_distr = torch.distributions.Normal(
            loc=torch.tensor([0.0]*n_experts, device=device), 
            scale=torch.tensor([1.0/n_experts]*n_experts, device=device)
        )
        
        # 原始obs处理分支（保留）
        if self.preprocess:
            self.obs_net = mlp(obs_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            if not add_trunk:
                self.trunk = nn.Identity()
                feature_dim = 2 * feature_dim
            else:
                self.trunk = mlp(2 * feature_dim, hidden_dim, "irelu")
                feature_dim = hidden_dim
        else:
            self.trunk = mlp(obs_dim + z_dim, hidden_dim, "ntanh",
                            hidden_dim, "irelu",
                            hidden_dim, "irelu")
            feature_dim = hidden_dim

        self.policy = mlp(feature_dim, hidden_dim, "irelu", action_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, z, std, router_noise=False):
        assert z.shape[-1] == self.z_dim

        if self.preprocess:
            # MOE处理obs_z分支
            obs_z = torch.cat([obs, z], dim=-1)
            router_logits = self.router(obs_z)
            if router_noise:
                # 添加噪声促进探索
                noisy_logits = router_logits + self.noise_distr.sample()
                router_logits = noisy_logits
                self.z_loss = (1 / obs_z.shape[0]) * torch.square(torch.exp(router_logits).sum(1)).sum(0)
            
            router_probs = F.softmax(router_logits, dim=-1)
            
            # Top-k稀疏化
            topk_probs, topk_indices = torch.topk(router_probs, self.topk, dim=-1)
            sparse_probs = torch.zeros_like(router_probs).scatter(
                dim=-1, index=topk_indices, src=topk_probs
            )
            
            # 专家组合
            expert_outputs = torch.stack([expert(obs_z) for expert in self.experts], dim=1)
            obs_z = torch.sum(expert_outputs * sparse_probs.unsqueeze(-1), dim=1)

            # 原始obs分支
            obs = self.obs_net(obs)
            h = torch.cat([obs, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z], dim=-1)
        
        if hasattr(self, "trunk"):
            h = self.trunk(h)
        
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        return utils.TruncatedNormal(mu, std)

class MOEActor2(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim,
                 n_experts=6, topk=2, preprocess=False, add_trunk=True, device='cuda') -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.preprocess = preprocess
        self.n_experts = n_experts
        self.topk = topk

        # MOE components for obs_z processing
        # self.router = nn.Linear(obs_dim + z_dim, n_experts)  # 路由网络
        self.router = nn.Sequential(nn.Linear(z_dim, int(hidden_dim/2)),
                                    nn.ReLU(),
                                    nn.Linear(int(hidden_dim/2), n_experts))
        self.experts = nn.ModuleList([
            mlp(obs_dim + z_dim, int(hidden_dim/2), "ntanh", feature_dim, "irelu")
            for _ in range(n_experts)
        ])
        
        # 噪声分布用于路由器探索
        self.noise_distr = torch.distributions.Normal(
            loc=torch.tensor([0.0]*n_experts, device=device), 
            scale=torch.tensor([1.0/n_experts]*n_experts, device=device)
        )
        
        # 原始obs处理分支（保留）
        if self.preprocess:
            self.obs_net = mlp(obs_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            if not add_trunk:
                self.trunk = nn.Identity()
                feature_dim = 2 * feature_dim
            else:
                self.trunk = mlp(2 * feature_dim, hidden_dim, "irelu")
                feature_dim = hidden_dim
        else:
            self.trunk = mlp(obs_dim + z_dim, hidden_dim, "ntanh",
                            hidden_dim, "irelu",
                            hidden_dim, "irelu")
            feature_dim = hidden_dim

        self.policy = mlp(feature_dim, hidden_dim, "irelu", action_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, z, std, router_noise=False):
        assert z.shape[-1] == self.z_dim

        if self.preprocess:
            # MOE处理obs_z分支
            obs_z = torch.cat([obs, z], dim=-1)
            router_logits = self.router(z)
            if router_noise:
                # 添加噪声促进探索
                noisy_logits = router_logits + self.noise_distr.sample()
                router_logits = noisy_logits
                self.z_loss = (1 / obs_z.shape[0]) * torch.square(torch.exp(router_logits).sum(1)).sum(0)
            
            router_probs = F.softmax(router_logits, dim=-1)
            
            # Top-k稀疏化
            topk_probs, topk_indices = torch.topk(router_probs, self.topk, dim=-1)
            sparse_probs = torch.zeros_like(router_probs).scatter(
                dim=-1, index=topk_indices, src=topk_probs
            )
            
            # 专家组合
            expert_outputs = torch.stack([expert(obs_z) for expert in self.experts], dim=1)
            obs_z = torch.sum(expert_outputs * sparse_probs.unsqueeze(-1), dim=1)

            # 原始obs分支
            obs = self.obs_net(obs)
            h = torch.cat([obs, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z], dim=-1)
        
        if hasattr(self, "trunk"):
            h = self.trunk(h)
        
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        return utils.TruncatedNormal(mu, std)
    
class DiagGaussianActor(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, hidden_dim, log_std_bounds,
                 preprocess=False) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.log_std_bounds = log_std_bounds
        self.preprocess = preprocess
        feature_dim = obs_dim + z_dim

        self.policy = mlp(feature_dim, hidden_dim, "ntanh", hidden_dim, "relu", 2 * action_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, z):
        assert z.shape[-1] == self.z_dim
        h = torch.cat([obs, z], dim=-1)
        mu, log_std = self.policy(h).chunk(2, dim=-1)
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()
        dist = utils.SquashedNormal(mu, std)
        return dist


class ForwardMap(nn.Module):
    """ forward representation class"""

    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim,
                 preprocess=False, add_trunk=True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.preprocess = preprocess

        if self.preprocess:
            self.obs_action_net = mlp(self.obs_dim + self.action_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            self.obs_z_net = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            if not add_trunk:
                self.trunk: nn.Module = nn.Identity()
                feature_dim = 2 * feature_dim
            else:
                self.trunk = mlp(2 * feature_dim, hidden_dim, "irelu")
                feature_dim = hidden_dim
        else:
            self.trunk = mlp(self.obs_dim + self.z_dim + self.action_dim, hidden_dim, "ntanh",
                             hidden_dim, "irelu",
                             hidden_dim, "irelu")
            feature_dim = hidden_dim

        seq = [feature_dim, hidden_dim, "irelu", self.z_dim]
        self.F1 = mlp(*seq)
        self.F2 = mlp(*seq)

        self.apply(utils.weight_init)

    def forward(self, obs, z, action):
        assert z.shape[-1] == self.z_dim

        if self.preprocess:
            obs_action = self.obs_action_net(torch.cat([obs, action], dim=-1))
            obs_z = self.obs_z_net(torch.cat([obs, z], dim=-1))
            h = torch.cat([obs_action, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z, action], dim=-1)
        if hasattr(self, "trunk"):
            h = self.trunk(h)
        F1 = self.F1(h)
        F2 = self.F2(h)
        return F1, F2
    
class ForwardMap_woz(nn.Module):
    """ forward representation class"""

    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim,
                 preprocess=False, add_trunk=True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.preprocess = preprocess

        if self.preprocess:
            self.obs_action_net = mlp(self.obs_dim + self.action_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            self.obs_z_net = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            if not add_trunk:
                self.trunk: nn.Module = nn.Identity()
                feature_dim = feature_dim
            else:
                self.trunk = mlp(feature_dim, hidden_dim, "irelu")
                feature_dim = hidden_dim
        else:
            self.trunk = mlp(self.obs_dim + self.action_dim, hidden_dim, "ntanh",
                             hidden_dim, "irelu",
                             hidden_dim, "irelu")
            feature_dim = hidden_dim

        seq = [feature_dim, hidden_dim, "irelu", self.z_dim]
        self.F1 = mlp(*seq)
        self.F2 = mlp(*seq)

        self.apply(utils.weight_init)

    def forward(self, obs, z, action):
        assert z.shape[-1] == self.z_dim

        if self.preprocess:
            obs_action = self.obs_action_net(torch.cat([obs, action], dim=-1))
            h = obs_action
        else:
            h = torch.cat([obs, action], dim=-1)
        if hasattr(self, "trunk"):
            h = self.trunk(h)
        F1 = self.F1(h)
        F2 = self.F2(h)
        return F1, F2

class MOEForwardMap(nn.Module):
    """ MOE version of forward representation class"""

    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim,
                 preprocess=False, add_trunk=True, n_experts=8, topk=3, device='cuda') -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.preprocess = preprocess
        self.n_experts = n_experts
        self.topk = topk

        # 噪声分布用于路由器探索
        self.noise_distr = torch.distributions.Normal(
            loc=torch.tensor([0.0]*n_experts, device=device), 
            scale=torch.tensor([1.0/n_experts]*n_experts, device=device)
        )

        # 预处理网络保持不变
        if self.preprocess:
            self.obs_action_net = mlp(self.obs_dim + self.action_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            self.obs_z_net = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            if not add_trunk:
                self.trunk: nn.Module = nn.Identity()
                trunk_output_dim = 2 * feature_dim
            else:
                self.trunk = mlp(2 * feature_dim, hidden_dim, "irelu")
                trunk_output_dim = hidden_dim
        else:
            self.trunk = mlp(self.obs_dim + self.z_dim + self.action_dim, hidden_dim, "ntanh",
                           hidden_dim, "irelu",
                           hidden_dim, "irelu")
            trunk_output_dim = hidden_dim

        # MOE组件 - 两个独立的路由器分别对应F1和F2
        self.router_F1 = nn.Sequential(nn.Linear(trunk_output_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, n_experts))
        self.router_F2 = nn.Sequential(nn.Linear(trunk_output_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, n_experts))
        
        # 专家网络 - 每个F1和F2都有自己的一组专家
        self.F1_experts = nn.ModuleList([
            mlp(trunk_output_dim, hidden_dim, "irelu", self.z_dim) 
            for _ in range(n_experts)
        ])
        
        self.F2_experts = nn.ModuleList([
            mlp(trunk_output_dim, hidden_dim, "irelu", self.z_dim)
            for _ in range(n_experts)
        ])
        
        self.apply(utils.weight_init)
        # 初始化路由器权重
        # nn.init.zeros_(self.router_F1.weight)
        # nn.init.zeros_(self.router_F2.weight)
        # 初始化最后一个专家层权重为零
        for expert in self.F1_experts + self.F2_experts:
            expert[-1].weight.data.fill_(0.0)

        # 专家使用统计
        self.register_buffer('F1_expert_usage', torch.zeros(n_experts))
        self.register_buffer('F2_expert_usage', torch.zeros(n_experts))

    def forward(self, obs, z, action, router_noise=False):
        assert z.shape[-1] == self.z_dim

        # 预处理阶段保持不变
        if self.preprocess:
            obs_action = self.obs_action_net(torch.cat([obs, action], dim=-1))
            obs_z = self.obs_z_net(torch.cat([obs, z], dim=-1))
            h = torch.cat([obs_action, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z, action], dim=-1)
        
        if hasattr(self, "trunk"):
            h = self.trunk(h)

        # F1的MOE路由逻辑
        F1, z_loss1 = self._moe_forward(h, self.router_F1, self.F1_experts, router_noise, self.F1_expert_usage)
        
        # F2的MOE路由逻辑
        F2, z_loss2 = self._moe_forward(h, self.router_F2, self.F2_experts, router_noise, self.F2_expert_usage)
        
        self.z_loss = z_loss1 + z_loss2
        
        return F1, F2

    def _moe_forward(self, h, router, experts, router_noise, usage_counter):
        """通用的MOE前向计算"""
        router_logits = router(h)
        
        z_loss = 0

        if router_noise:
            # 添加噪声促进探索
            router_logits = router_logits + self.noise_distr.sample().to(h.device)
            
            z_loss = (1 / h.shape[0]) * torch.square(torch.exp(router_logits).sum(1)).sum(0)
            
            # 计算路由器重要性指标
            with torch.no_grad():
                importance = F.softmax(router_logits, dim=-1).sum(0)
                self.router_importance = (torch.std(importance)/torch.mean(importance))**2
                
                # 计算路由器负载指标
                threshold = torch.max(router_logits, dim=-1).values
                load = (1 - self.noise_distr.cdf(threshold.unsqueeze(1) - router_logits)).sum(0)
                self.router_load = (torch.std(load)/torch.mean(load))**2

        # 获取topk专家
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, self.topk, dim=-1)
        sparse_probs = torch.zeros_like(router_probs).scatter_(
            index=topk_indices, src=topk_probs, dim=-1)
        
        # 更新专家使用统计
        if self.training:
            with torch.no_grad():
                expert_mask = torch.zeros_like(router_probs)
                expert_mask.scatter_(1, topk_indices, 1)
                usage_counter += expert_mask.sum(0)

        # 并行计算所有专家输出并加权组合
        expert_outputs = torch.stack([expert(h) for expert in experts], dim=1)
        output = torch.sum(expert_outputs * sparse_probs.unsqueeze(-1), dim=1)
        
        return output, z_loss
    
class MOEForwardMap1(nn.Module):
    """ forward representation class"""

    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim,
                 preprocess=False, add_trunk=True, n_experts=8, topk=3, device='cuda') -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.preprocess = preprocess
        self.n_experts = n_experts
        self.topk = topk
        
        # MOE components for obs_z processing
        # self.router = nn.Linear(obs_dim + z_dim, n_experts)  # 路由网络
        self.router = nn.Sequential(nn.Linear(obs_dim + z_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, n_experts))
        
        self.experts = nn.ModuleList([
            mlp(obs_dim + z_dim, int(hidden_dim/2), "ntanh", feature_dim, "irelu")
            for _ in range(n_experts)
        ])
        
        # 噪声分布用于路由器探索
        self.noise_distr = torch.distributions.Normal(
            loc=torch.tensor([0.0]*n_experts, device=device), 
            scale=torch.tensor([1.0/n_experts]*n_experts, device=device)
        )
        
        if self.preprocess:
            self.obs_action_net = mlp(self.obs_dim + self.action_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            # self.obs_z_net = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            if not add_trunk:
                self.trunk: nn.Module = nn.Identity()
                feature_dim = 2 * feature_dim
            else:
                self.trunk = mlp(2 * feature_dim, hidden_dim, "irelu")
                feature_dim = hidden_dim
        else:
            self.trunk = mlp(self.obs_dim + self.z_dim + self.action_dim, hidden_dim, "ntanh",
                             hidden_dim, "irelu",
                             hidden_dim, "irelu")
            feature_dim = hidden_dim

        seq = [feature_dim, hidden_dim, "irelu", self.z_dim]
        self.F1 = mlp(*seq)
        self.F2 = mlp(*seq)

        self.apply(utils.weight_init)

    def forward(self, obs, z, action, router_noise=False):
        assert z.shape[-1] == self.z_dim

        if self.preprocess:
            # MOE处理obs_z分支
            obs_z = torch.cat([obs, z], dim=-1)
            router_logits = self.router(obs_z)
            if router_noise:
                # 添加噪声促进探索
                noisy_logits = router_logits + self.noise_distr.sample()
                router_logits = noisy_logits
                self.z_loss = (1 / obs_z.shape[0]) * torch.square(torch.exp(router_logits).sum(1)).sum(0)
            
            router_probs = F.softmax(router_logits, dim=-1)
            
            # Top-k稀疏化
            topk_probs, topk_indices = torch.topk(router_probs, self.topk, dim=-1)
            sparse_probs = torch.zeros_like(router_probs).scatter(
                dim=-1, index=topk_indices, src=topk_probs
            )
            
            # 专家组合
            expert_outputs = torch.stack([expert(obs_z) for expert in self.experts], dim=1)
            obs_z = torch.sum(expert_outputs * sparse_probs.unsqueeze(-1), dim=1)

            # 原始obs分支
            obs_action = self.obs_action_net(torch.cat([obs, action], dim=-1))
            # obs_z = self.obs_z_net(torch.cat([obs, z], dim=-1))
            h = torch.cat([obs_action, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z, action], dim=-1)
        if hasattr(self, "trunk"):
            h = self.trunk(h)
        F1 = self.F1(h)
        F2 = self.F2(h)
        return F1, F2

class IdentityMap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.B = nn.Identity()

    def forward(self, obs):
        return self.B(obs)


class BackwardMap(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.B = mlp(self.obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs):
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        B = self.B(obs)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=-1)
        return B

# 注意力池化
class AttentionPooling(nn.Module):
    def __init__(self, z_dim):
        super(AttentionPooling, self).__init__()
        self.z_dim = z_dim
        self.query_vector = nn.Parameter(torch.randn(z_dim))  # 可学习的查询向量

    def forward(self, x):
        # x: (batch_size, seq_len, z_dim)
        # 计算注意力权重
        attention_scores = torch.matmul(x, self.query_vector) / (self.z_dim ** 0.5)  # (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len)

        # 加权求和
        output = torch.matmul(attention_weights.unsqueeze(1), x)  # (batch_size, 1, z_dim)
        output = output.squeeze(1)  # (batch_size, z_dim)
        return output
    
# 注意力池化attention
class PureAttention(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z):
        super(PureAttention, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.norm_z = norm_z
        # 注意力池化
        self.attention_pooling = AttentionPooling(z_dim)
        # 可学习的注意力向量
        self.embed = mlp(self.obs_dim, hidden_dim, "ntanh")
        self.query = mlp(hidden_dim, "relu", self.z_dim)
        self.key = mlp(hidden_dim, "relu", self.z_dim)
        self.value = mlp(hidden_dim, "relu", self.z_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_size)
        embed = self.embed(x)
        Q = self.query(embed)  # (batch_size, seq_len, embed_size)
        K = self.key(embed)    # (batch_size, seq_len, embed_size)
        V = self.value(embed)  # (batch_size, seq_len, embed_size)

        # 计算注意力权重
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.z_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 加权求和
        output = torch.matmul(attention_weights, V)  # (batch_size, seq_len, embed_size)
        # 注意力池化
        pooled_output = self.attention_pooling(output)  # (batch_size, z_dim)
        # 归一化
        if self.norm_z:
            output = math.sqrt(self.z_dim) * F.normalize(pooled_output, dim=-1)
        
        return output

# 稀疏注意力模型
class SparseAttention(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z, window_size=10):
        super(SparseAttention, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.norm_z = norm_z
        self.window_size = window_size  # 局部注意力窗口大小

        # 注意力池化
        self.attention_pooling = AttentionPooling(z_dim)
        # 可学习的注意力向量
        self.embed = mlp(self.obs_dim, hidden_dim, "ntanh")
        self.query = mlp(hidden_dim, "relu", self.z_dim)
        self.key = mlp(hidden_dim, "relu", self.z_dim)
        self.value = mlp(hidden_dim, "relu", self.z_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_size)
        batch_size, seq_len, _ = x.shape
        embed = self.embed(x)
        Q = self.query(embed)  # (batch_size, seq_len, embed_size)
        K = self.key(embed)    # (batch_size, seq_len, embed_size)
        V = self.value(embed)  # (batch_size, seq_len, embed_size)

        # 初始化输出
        output = torch.zeros_like(Q)

        # 局部注意力计算
        for i in range(seq_len):
            # 确定局部窗口的范围
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)

            # 计算局部注意力权重
            local_Q = Q[:, i:i+1, :]  # (batch_size, 1, embed_size)
            local_K = K[:, start:end, :]  # (batch_size, window_size, embed_size)
            local_V = V[:, start:end, :]  # (batch_size, window_size, embed_size)

            # 计算局部注意力分数
            local_attention_scores = torch.matmul(local_Q, local_K.transpose(-2, -1)) / (self.z_dim ** 0.5)
            local_attention_weights = F.softmax(local_attention_scores, dim=-1)

            # 加权求和
            local_output = torch.matmul(local_attention_weights, local_V)  # (batch_size, 1, embed_size)
            output[:, i:i+1, :] = local_output

        # 注意力池化
        pooled_output = self.attention_pooling(output)  # (batch_size, z_dim)
        # 归一化
        if self.norm_z:
            output = math.sqrt(self.z_dim) * F.normalize(pooled_output, dim=-1)
        
        return output
    
# CLS token attention
class PureAttentionWithCLSToken(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z):
        super(PureAttentionWithCLSToken, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.norm_z = norm_z
        # 可学习的 CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, obs_dim))  # (1, 1, obs_dim)
        # 可学习的注意力向量
        self.embed = mlp(self.obs_dim, hidden_dim, "ntanh")
        self.query = mlp(hidden_dim, "relu", self.z_dim)
        self.key = mlp(hidden_dim, "relu", self.z_dim)
        self.value = mlp(hidden_dim, "relu", self.z_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, obs_dim)
        batch_size = x.size(0)
        # 添加 CLS Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, obs_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, seq_len + 1, obs_dim)

        # 嵌入层
        embed = self.embed(x)
        Q = self.query(embed)  # (batch_size, seq_len + 1, z_dim)
        K = self.key(embed)    # (batch_size, seq_len + 1, z_dim)
        V = self.value(embed)  # (batch_size, seq_len + 1, z_dim)

        # 计算注意力权重
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.z_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 加权求和
        output = torch.matmul(attention_weights, V)  # (batch_size, seq_len + 1, z_dim)
        if self.norm_z:
            output = math.sqrt(self.z_dim) * F.normalize(output, dim=-1)

        # 取 CLS Token 对应的输出
        cls_output = output[:, 0, :]  # (batch_size, z_dim)
        return cls_output

# 稀疏注意力模型（带 CLS Token）
class SparseAttentionWithCLSToken(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z, window_size=10):
        super(SparseAttentionWithCLSToken, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.norm_z = norm_z
        self.window_size = window_size  # 局部注意力窗口大小

        # 可学习的 CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, obs_dim))  # (1, 1, obs_dim)
        # 可学习的注意力向量
        self.embed = mlp(self.obs_dim, hidden_dim, "ntanh")
        self.query = mlp(hidden_dim, "relu", self.z_dim)
        self.key = mlp(hidden_dim, "relu", self.z_dim)
        self.value = mlp(hidden_dim, "relu", self.z_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, obs_dim)
        batch_size, seq_len, _ = x.shape
        # 添加 CLS Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, obs_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, seq_len + 1, obs_dim)

        # 嵌入层
        embed = self.embed(x)
        Q = self.query(embed)  # (batch_size, seq_len + 1, z_dim)
        K = self.key(embed)    # (batch_size, seq_len + 1, z_dim)
        V = self.value(embed)  # (batch_size, seq_len + 1, z_dim)

        # 初始化输出
        output = torch.zeros_like(Q)

        # 局部注意力计算
        for i in range(seq_len + 1):  # 包括 CLS Token
            # 确定局部窗口的范围
            start = max(0, i - self.window_size // 2)
            end = min(seq_len + 1, i + self.window_size // 2 + 1)

            # 计算局部注意力权重
            local_Q = Q[:, i:i+1, :]  # (batch_size, 1, z_dim)
            local_K = K[:, start:end, :]  # (batch_size, window_size, z_dim)
            local_V = V[:, start:end, :]  # (batch_size, window_size, z_dim)

            # 计算局部注意力分数
            local_attention_scores = torch.matmul(local_Q, local_K.transpose(-2, -1)) / (self.z_dim ** 0.5)
            local_attention_weights = F.softmax(local_attention_scores, dim=-1)

            # 加权求和
            local_output = torch.matmul(local_attention_weights, local_V)  # (batch_size, 1, z_dim)
            output[:, i:i+1, :] = local_output

        # 归一化
        if self.norm_z:
            output = math.sqrt(self.z_dim) * F.normalize(output, dim=-1)

        # 取 CLS Token 对应的输出
        cls_output = output[:, 0, :]  # (batch_size, z_dim)
        return cls_output

# 稀疏注意力模型
class SparseAttention1(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z, window_size=10):
        super(SparseAttention1, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.norm_z = norm_z
        self.window_size = window_size  # 局部注意力窗口大小

        # 注意力池化
        self.attention_pooling = AttentionPooling(z_dim)
        # 可学习的注意力向量
        self.embed = mlp(self.obs_dim, hidden_dim, "ntanh")
        self.query = mlp(hidden_dim, "relu", self.z_dim)
        self.key = mlp(hidden_dim, "relu", self.z_dim)
        self.value = mlp(hidden_dim, "relu", self.z_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_size)
        batch_size, seq_len, _ = x.shape
        embed = self.embed(x)
        Q = self.query(embed)  # (batch_size, seq_len, embed_size)
        K = self.key(embed)    # (batch_size, seq_len, embed_size)
        V = self.value(embed)  # (batch_size, seq_len, embed_size)

        # 初始化输出
        output = torch.zeros_like(Q)

        # 局部注意力计算
        for i in range(seq_len):
            # 确定局部窗口的范围
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)

            # 计算局部注意力权重
            local_Q = Q[:, i:i+1, :]  # (batch_size, 1, embed_size)
            local_K = K[:, start:end, :]  # (batch_size, window_size, embed_size)
            local_V = V[:, start:end, :]  # (batch_size, window_size, embed_size)
            
            if self.norm_z:
                local_Q = math.sqrt(self.z_dim) * F.normalize(local_Q, dim=-1)
                local_K = math.sqrt(self.z_dim) * F.normalize(local_K, dim=-1)
                local_V = math.sqrt(self.z_dim) * F.normalize(local_V, dim=-1)

            # 计算局部注意力分数
            local_attention_scores = torch.matmul(local_Q, local_K.transpose(-2, -1)) / (self.z_dim ** 0.5)
            local_attention_weights = F.softmax(local_attention_scores, dim=-1)

            # 加权求和
            local_output = torch.matmul(local_attention_weights, local_V)  # (batch_size, 1, embed_size)
            output[:, i:i+1, :] = local_output

        # 注意力池化
        pooled_output = self.attention_pooling(output)  # (batch_size, z_dim)
        # 归一化
        if self.norm_z:
            output = math.sqrt(self.z_dim) * F.normalize(pooled_output, dim=-1)
        
        return output

# 稀疏注意力模型（带 CLS Token）
class SparseAttentionWithCLSToken1(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z, window_size=10):
        super(SparseAttentionWithCLSToken1, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.norm_z = norm_z
        self.window_size = window_size  # 局部注意力窗口大小

        # 可学习的 CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, obs_dim))  # (1, 1, obs_dim)
        # 可学习的注意力向量
        self.embed = mlp(self.obs_dim, hidden_dim, "ntanh")
        self.query = mlp(hidden_dim, "relu", self.z_dim)
        self.key = mlp(hidden_dim, "relu", self.z_dim)
        self.value = mlp(hidden_dim, "relu", self.z_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, obs_dim)
        batch_size, seq_len, _ = x.shape
        # 添加 CLS Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, obs_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, seq_len + 1, obs_dim)

        # 嵌入层
        embed = self.embed(x)
        Q = self.query(embed)  # (batch_size, seq_len + 1, z_dim)
        K = self.key(embed)    # (batch_size, seq_len + 1, z_dim)
        V = self.value(embed)  # (batch_size, seq_len + 1, z_dim)

        # 初始化输出
        output = torch.zeros_like(Q)

        # 局部注意力计算
        for i in range(seq_len + 1):  # 包括 CLS Token
            # 确定局部窗口的范围
            start = max(0, i - self.window_size // 2)
            end = min(seq_len + 1, i + self.window_size // 2 + 1)

            # 计算局部注意力权重
            local_Q = Q[:, i:i+1, :]  # (batch_size, 1, z_dim)
            local_K = K[:, start:end, :]  # (batch_size, window_size, z_dim)
            local_V = V[:, start:end, :]  # (batch_size, window_size, z_dim)
            
            if self.norm_z:
                local_Q = math.sqrt(self.z_dim) * F.normalize(local_Q, dim=-1)
                local_K = math.sqrt(self.z_dim) * F.normalize(local_K, dim=-1)
                local_V = math.sqrt(self.z_dim) * F.normalize(local_V, dim=-1)

            # 计算局部注意力分数
            local_attention_scores = torch.matmul(local_Q, local_K.transpose(-2, -1)) / (self.z_dim ** 0.5)
            local_attention_weights = F.softmax(local_attention_scores, dim=-1)

            # 加权求和
            local_output = torch.matmul(local_attention_weights, local_V)  # (batch_size, 1, z_dim)
            output[:, i:i+1, :] = local_output

        # 归一化
        if self.norm_z:
            output = math.sqrt(self.z_dim) * F.normalize(output, dim=-1)

        # 取 CLS Token 对应的输出
        cls_output = output[:, 0, :]  # (batch_size, z_dim)
        return cls_output
    
# 注意力池化attention
class PureAttention1(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z):
        super(PureAttention1, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.norm_z = norm_z
        # 注意力池化
        self.attention_pooling = AttentionPooling(z_dim)
        # 可学习的注意力向量
        self.embed = mlp(self.obs_dim, hidden_dim, "ntanh")
        self.query = mlp(hidden_dim, "relu", self.z_dim)
        self.key = mlp(hidden_dim, "relu", self.z_dim)
        self.value = mlp(hidden_dim, "relu", self.z_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_size)
        embed = self.embed(x)
        Q = self.query(embed)  # (batch_size, seq_len, embed_size)
        K = self.key(embed)    # (batch_size, seq_len, embed_size)
        V = self.value(embed)  # (batch_size, seq_len, embed_size)

        if self.norm_z:
            Q = math.sqrt(self.z_dim) * F.normalize(Q, dim=-1)
            K = math.sqrt(self.z_dim) * F.normalize(K, dim=-1)
            V = math.sqrt(self.z_dim) * F.normalize(V, dim=-1)

        # 计算注意力权重
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.z_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 加权求和
        output = torch.matmul(attention_weights, V)  # (batch_size, seq_len, embed_size)
        # 注意力池化
        pooled_output = self.attention_pooling(output)  # (batch_size, z_dim)
        # 归一化
        if self.norm_z:
            output = math.sqrt(self.z_dim) * F.normalize(pooled_output, dim=-1)
        
        return output
    
# CLS token attention
class PureAttentionWithCLSToken1(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z):
        super(PureAttentionWithCLSToken1, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.norm_z = norm_z
        # 可学习的 CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, obs_dim))  # (1, 1, obs_dim)
        # 可学习的注意力向量
        self.embed = mlp(self.obs_dim, hidden_dim, "ntanh")
        self.query = mlp(hidden_dim, "relu", self.z_dim)
        self.key = mlp(hidden_dim, "relu", self.z_dim)
        self.value = mlp(hidden_dim, "relu", self.z_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, obs_dim)
        batch_size = x.size(0)
        # 添加 CLS Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, obs_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, seq_len + 1, obs_dim)

        # 嵌入层
        embed = self.embed(x)
        Q = self.query(embed)  # (batch_size, seq_len + 1, z_dim)
        K = self.key(embed)    # (batch_size, seq_len + 1, z_dim)
        V = self.value(embed)  # (batch_size, seq_len + 1, z_dim)
        
        if self.norm_z:
            Q = math.sqrt(self.z_dim) * F.normalize(Q, dim=-1)
            K = math.sqrt(self.z_dim) * F.normalize(K, dim=-1)
            V = math.sqrt(self.z_dim) * F.normalize(V, dim=-1)

        # 计算注意力权重
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.z_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 加权求和
        output = torch.matmul(attention_weights, V)  # (batch_size, seq_len + 1, z_dim)
        if self.norm_z:
            output = math.sqrt(self.z_dim) * F.normalize(output, dim=-1)

        # 取 CLS Token 对应的输出
        cls_output = output[:, 0, :]  # (batch_size, z_dim)
        return cls_output
    
# 输入s和s'的版本
class BackwardMap2(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.B = mlp(2*self.obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        inpt = torch.cat([obs, next_obs], dim=-1)
        B = self.B(inpt)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=-1)
        return B
    
class MultinputNet(nn.Module):
    """Network with multiple inputs"""

    def __init__(self, input_dims: tp.Sequence[int], sequence_dims: tp.Sequence[int]) -> None:
        super().__init__()
        input_dims = list(input_dims)
        sequence_dims = list(sequence_dims)
        dim0 = sequence_dims[0]
        self.innets = nn.ModuleList([mlp(indim, dim0, "relu", dim0, "layernorm") for indim in input_dims])  # type: ignore
        sequence: tp.List[tp.Union[str, int]] = [dim0]
        for dim in sequence_dims[1:]:
            sequence.extend(["relu", dim])
        self.outnet = mlp(*sequence)  # type: ignore

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        assert len(tensors) == len(self.innets)
        out = sum(net(x) for net, x in zip(self.innets, tensors)) / len(self.innets)
        return self.outnet(out)  # type : ignore
