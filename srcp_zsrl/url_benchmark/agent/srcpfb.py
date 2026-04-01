import copy
import math
import logging
import dataclasses
from collections import OrderedDict
import typing as tp
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
import omegaconf
import os
from url_benchmark import utils
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark.dmc import TimeStep
from .ddpg import MetaDict
from .cm_networks1 import Seperate_CPActor6 as Actor
from .fb_modules import IdentityMap
from .fb_modules import DiagGaussianActor, ForwardMap, BackwardMap
from torchvision.utils import save_image
from utils import fb_compute_attribution, compute_attribution_mask, make_attribution_pred_grid, make_obs_grid, make_obs_grad_grid

logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        # self.repr_dim = 32 * 35 * 35
        final_shape = int(obs_shape[1] / 2 -7)
        self.out_dim = 32 * final_shape * final_shape
        self.repr_dim = 512

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.trunk = nn.Sequential(nn.Linear(self.out_dim, self.repr_dim),
                                   nn.LayerNorm(self.repr_dim), nn.Tanh())
        
        self.apply(utils.weight_init)
        # # Initialization
        # nn.init.orthogonal_(self.fc.weight.data)
        # self.fc.bias.data.fill_(0.0)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        out = self.trunk(h)
        return out

class ICM(nn.Module):
    """
    Same as ICM, with a trunk to save memory for KNN
    """
    def __init__(self, obs_dim, action_dim, hidden_dim, icm_rep_dim):
        super().__init__()

        self.forward_net = nn.Sequential(
            nn.Linear(icm_rep_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, icm_rep_dim))

        self.backward_net = nn.Sequential(
            nn.Linear(2 * icm_rep_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs, action, next_obs):
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        next_obs_hat = self.forward_net(torch.cat([obs, action], dim=-1))
        action_hat = self.backward_net(torch.cat([obs, next_obs], dim=-1))

        forward_error = torch.norm(next_obs - next_obs_hat,
                                   dim=-1,
                                   p=2,
                                   keepdim=True)
        backward_error = torch.norm(action - action_hat,
                                    dim=-1,
                                    p=2,
                                    keepdim=True)

        return forward_error, backward_error

    def get_rep(self, obs, action):
        return obs
    
def make_aug_encoder(cfg):
    aug = utils.RandomShiftsAug(pad=(cfg.image_wh // 21))
    encoder = Encoder(cfg.obs_shape).to(cfg.device)
    example_ob = torch.zeros(1, *cfg.obs_shape, device=cfg.device)
    module_obs_dim = encoder(example_ob).shape[-1]
    encoder.repr_dim = module_obs_dim

    return aug, encoder

@dataclasses.dataclass
class SRCPFBAgentConfig:
    # @package agent
    _target_: str = "url_benchmark.agent.srcpfb.SRCPFBAgent"
    name: str = "srcpfb"
    obs_type: str = omegaconf.MISSING  # to be specified later
    image_wh: int = omegaconf.MISSING  # to be specified later
    obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    device: str = omegaconf.II("device")  # ${device}
    domain = omegaconf.MISSING
    seed = omegaconf.MISSING
    lr: float = 1e-4
    lr_coef: float = 1
    fb_target_tau: float = 0.01  # 0.001-0.01
    update_every_steps: int = 1
    use_tb: bool = omegaconf.II("use_tb")  # ${use_tb}
    use_wandb: bool = omegaconf.II("use_wandb")  # ${use_wandb}
    num_expl_steps: int = omegaconf.MISSING  # ???  # to be specified later
    num_inference_steps: int = 10000
    hidden_dim: int = 1024   # 128, 2048
    backward_hidden_dim: int = 512   # 512
    feature_dim: int = 512   # 128, 1024
    z_dim: int = 50  # 100
    stddev_schedule: str = "0.2"  # "linear(1,0.2,200000)" #
    stddev_clip: float = 0.3  # 1
    update_z_every_step: int = 300
    update_z_proba: float = 1.0
    nstep: int = 1
    batch_size: int = 1024  # 512
    init_fb: bool = True
    update_encoder: bool = omegaconf.II("update_encoder")  # ${update_encoder}
    ortho_coef: float = 1.0  # 0.01-10
    log_std_bounds: tp.Tuple[float, float] = (-5, 2)  # param for DiagGaussianActor
    temp: float = 1  # temperature for DiagGaussianActor
    boltzmann: bool = False  # set to true for DiagGaussianActor
    debug: bool = False
    future_ratio: float = 0.0
    mix_ratio: float = 0.5  # 0-1
    rand_weight: bool = False  # True, False
    preprocess: bool = True
    norm_z: bool = True
    q_loss: bool = False
    q_loss_coef: float = 0.01
    additional_metric: bool = False
    add_trunk: bool = False


cs = ConfigStore.instance()
cs.store(group="agent", name="srcpfb", node=SRCPFBAgentConfig)


class SRCPFBAgent:

    # pylint: disable=unused-argument
    def __init__(self,
                 **kwargs: tp.Any
                 ):
        cfg = SRCPFBAgentConfig(**kwargs)
        self.cfg = cfg
        assert len(cfg.action_shape) == 1
        self.action_dim = cfg.action_shape[0]
        self.solved_meta: tp.Any = None
        
        print("cfg.device", cfg.device)

        # models
        if cfg.obs_type == 'pixels':
            self.aug, self.encoder = make_aug_encoder(cfg)
            self.obs_dim = self.encoder.repr_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = cfg.obs_shape[0]
        # create the network
        if self.cfg.boltzmann:
            self.actor: nn.Module = DiagGaussianActor(cfg.obs_type, self.obs_dim, cfg.z_dim, self.action_dim,
                                                      cfg.hidden_dim, cfg.log_std_bounds).to(cfg.device)
            self.actor_target: nn.Module = DiagGaussianActor(cfg.obs_type, self.obs_dim, cfg.z_dim, self.action_dim,
                                                      cfg.hidden_dim, cfg.log_std_bounds).to(cfg.device)
        else:
            self.actor = Actor(self.obs_dim, cfg.z_dim, self.action_dim, cfg.device, cfg.feature_dim, cfg.hidden_dim).to(cfg.device)
            self.actor_target = Actor(self.obs_dim, cfg.z_dim, self.action_dim, cfg.device, cfg.feature_dim, cfg.hidden_dim).to(cfg.device)
        self.forward_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
                                      cfg.feature_dim, cfg.hidden_dim,
                                      preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        if cfg.debug:
            self.backward_net: nn.Module = IdentityMap().to(cfg.device)
            self.backward_target_net: nn.Module = IdentityMap().to(cfg.device)
        else:
            self.backward_net = BackwardMap(self.obs_dim, cfg.z_dim, cfg.backward_hidden_dim, norm_z=cfg.norm_z).to(cfg.device)
            self.backward_target_net = BackwardMap(self.obs_dim,
                                                   cfg.z_dim, cfg.backward_hidden_dim, norm_z=cfg.norm_z).to(cfg.device)
        # build up the target network
        self.forward_target_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
                                             cfg.feature_dim, cfg.hidden_dim,
                                             preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        # load the weights into the target networks
        self.forward_target_net.load_state_dict(self.forward_net.state_dict())
        self.backward_target_net.load_state_dict(self.backward_net.state_dict())
        # optimizers
        self.encoder_opt: tp.Optional[torch.optim.Adam] = None
        if cfg.obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.fb_opt = torch.optim.Adam([{'params': self.forward_net.parameters()},  # type: ignore
                                        {'params': self.backward_net.parameters(), 'lr': cfg.lr_coef * cfg.lr}],
                                       lr=cfg.lr)

        self.train()
        self.forward_target_net.train()
        self.backward_target_net.train()
        self.actor_success: tp.List[float] = []  # only for debugging, can be removed eventually
        self.icm_rep_dim=self.encoder.repr_dim
        self.icm = ICM(self.obs_dim, self.action_dim, cfg.hidden_dim,
                       self.icm_rep_dim).to(cfg.device)
        
        # optimizers
        self.icm_opt = torch.optim.Adam(self.icm.parameters(), lr=cfg.lr)
        self.icm.train()
        self.quantile = 0.9
        
        self.domain = cfg.domain
        self.seed = cfg.seed
        self.work_dir = os.path.join(str(self.domain)+'_'+str(cfg.obs_type)+'_pretrain', str("srcpfb"), str(self.seed))
        self.save_dir = utils.make_dir(os.path.join(self.work_dir, 'save'))
        
    def train(self, training: bool = True) -> None:
        self.training = training
        for net in [self.encoder, self.actor, self.forward_net, self.backward_net]:
            net.train(training)

    def init_from(self, other) -> None:
        # copy parameters over
        names = ["encoder", "actor"]
        if self.cfg.init_fb:
            names += ["forward_net", "backward_net", "backward_target_net", "forward_target_net"]
        for name in names:
            utils.hard_update_params(getattr(other, name), getattr(self, name))
        for key, val in self.__dict__.items():
            if isinstance(val, torch.optim.Optimizer):
                val.load_state_dict(copy.deepcopy(getattr(other, key).state_dict()))

    def get_goal_meta(self, goal_array: np.ndarray, obs_array: np.ndarray = None) -> MetaDict:
        desired_goal = torch.tensor(goal_array, dtype=torch.float32).unsqueeze(0).to(self.cfg.device)
        with torch.no_grad():
            z = self.backward_net(desired_goal)
        if self.cfg.norm_z:
            z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
        z = z.squeeze(0).cpu().numpy()
        meta = OrderedDict()
        meta['z'] = z
        return meta

    def infer_meta_from_obs_and_rewards(self, obs: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor):
        with torch.no_grad():
            obs = self.encoder(obs)

        with torch.no_grad():
            B = self.backward_net(obs)
        z = torch.matmul(reward.T, B) / reward.shape[0]

        if self.cfg.norm_z:
            z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
        meta = OrderedDict()
        meta['z'] = z.squeeze().cpu().numpy()
        return meta

    def sample_z(self, size, device: str = "cpu"):
        gaussian_rdv = torch.randn((size, self.cfg.z_dim), dtype=torch.float32, device=device)
        gaussian_rdv = F.normalize(gaussian_rdv, dim=1)
        if self.cfg.norm_z:
            z = math.sqrt(self.cfg.z_dim) * gaussian_rdv
        else:
            uniform_rdv = torch.rand((size, self.cfg.z_dim), dtype=torch.float32, device=device)
            z = np.sqrt(self.cfg.z_dim) * uniform_rdv * gaussian_rdv
        return z

    def init_meta(self) -> MetaDict:
        if self.solved_meta is not None:
            print('solved_meta')
            return self.solved_meta
        else:
            z = self.sample_z(1)
            z = z.squeeze().numpy()
            meta = OrderedDict()
            meta['z'] = z
        return meta

    # pylint: disable=unused-argument
    def update_meta(
        self,
        meta: MetaDict,
        global_step: int,
        time_step: TimeStep,
        finetune: bool = False,
        replay_loader: tp.Optional[ReplayBuffer] = None
    ) -> MetaDict:
        if global_step % self.cfg.update_z_every_step == 0 and np.random.rand() < self.cfg.update_z_proba:
            return self.init_meta()
        return meta

    def act(self, obs, meta, step, eval_mode) -> tp.Any:
        obs = torch.as_tensor(obs, device=self.cfg.device, dtype=torch.float32).unsqueeze(0)  # type: ignore
        h = self.encoder(obs)
        z = torch.as_tensor(meta['z'], device=self.cfg.device).unsqueeze(0)  # type: ignore
        if self.cfg.boltzmann:
            dist = self.actor(h, z)
        else:
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            action = self.actor(h, z)
        if eval_mode:
            action = action
            if self.cfg.additional_metric:
                # the following is doing extra computation only used for metrics,
                # it should be deactivated eventually
                F_mean_s = self.forward_net(obs, z, action)
                F_rand_s = self.forward_net(obs, z, torch.zeros_like(action).uniform_(-1.0, 1.0))
                Qs = [torch.min(*(torch.einsum('sd, sd -> s', F, z) for F in Fs)) for Fs in [F_mean_s, F_rand_s]]
                self.actor_success = (Qs[0] > Qs[1]).cpu().numpy().tolist()
        else:
            action = action
            if step < self.cfg.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_fb(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        z: torch.Tensor,
        step: int
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        # compute target successor measure
        with torch.no_grad():
            if self.cfg.boltzmann:
                dist = self.actor(next_obs, z)
                next_action = dist.sample()
            else:
                stddev = utils.schedule(self.cfg.stddev_schedule, step)
                next_action = self.actor(next_obs, z)
            target_F1, target_F2 = self.forward_target_net(next_obs, z, next_action)  # batch x z_dim
            target_B = self.backward_target_net(next_obs)  # batch x z_dim
            target_M1 = torch.einsum('sd, td -> st', target_F1, target_B)  # batch x batch
            target_M2 = torch.einsum('sd, td -> st', target_F2, target_B)  # batch x batch
            target_M = torch.min(target_M1, target_M2)

        # compute FB loss
        F1, F2 = self.forward_net(obs, z, action)
        B = self.backward_net(next_obs)
        M1 = torch.einsum('sd, td -> st', F1, B)  # batch x batch
        M2 = torch.einsum('sd, td -> st', F2, B)  # batch x batch
        I = torch.eye(*M1.size(), device=M1.device)
        off_diag = ~I.bool()
        fb_offdiag: tp.Any = 0.5 * sum((M - discount * target_M)[off_diag].pow(2).mean() for M in [M1, M2])
        fb_diag: tp.Any = -sum(M.diag().mean() for M in [M1, M2])
        fb_loss = fb_offdiag + fb_diag

        # Q LOSS

        if self.cfg.q_loss:
            with torch.no_grad():
                next_Q1, nextQ2 = [torch.einsum('sd, sd -> s', target_Fi, z) for target_Fi in [target_F1, target_F2]]
                next_Q = torch.min(next_Q1, nextQ2)
                cov = torch.matmul(B.T, B) / B.shape[0]
                inv_cov = torch.inverse(cov)
                implicit_reward = (torch.matmul(B, inv_cov) * z).sum(dim=1)  # batch_size
                target_Q = implicit_reward.detach() + discount.squeeze(1) * next_Q  # batch_size
            Q1, Q2 = [torch.einsum('sd, sd -> s', Fi, z) for Fi in [F1, F2]]
            q_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
            fb_loss += self.cfg.q_loss_coef * q_loss

        # ORTHONORMALITY LOSS FOR BACKWARD EMBEDDING

        Cov = torch.matmul(B, B.T)
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss += self.cfg.ortho_coef * orth_loss

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['target_M'] = target_M.mean().item()
            metrics['M1'] = M1.mean().item()
            metrics['F1'] = F1.mean().item()
            metrics['B'] = B.mean().item()
            metrics['B_norm'] = torch.norm(B, dim=-1).mean().item()
            metrics['z_norm'] = torch.norm(z, dim=-1).mean().item()
            metrics['fb_loss'] = fb_loss.item()
            metrics['fb_diag'] = fb_diag.item()
            metrics['fb_offdiag'] = fb_offdiag.item()
            if self.cfg.q_loss:
                metrics['q_loss'] = q_loss.item()
            metrics['orth_loss'] = orth_loss.item()
            metrics['orth_loss_diag'] = orth_loss_diag.item()
            metrics['orth_loss_offdiag'] = orth_loss_offdiag.item()
            if self.cfg.q_loss:
                metrics['q_loss'] = q_loss.item()
            eye_diff = torch.matmul(B.T, B) / B.shape[0] - torch.eye(B.shape[1], device=B.device)
            metrics['orth_linf'] = torch.max(torch.abs(eye_diff)).item()
            metrics['orth_l2'] = eye_diff.norm().item() / math.sqrt(B.shape[1])
            if isinstance(self.fb_opt, torch.optim.Adam):
                metrics["fb_opt_lr"] = self.fb_opt.param_groups[0]["lr"]

        # optimize FB
        # if self.encoder_opt is not None:
        #     self.encoder_opt.zero_grad(set_to_none=True)
        self.fb_opt.zero_grad(set_to_none=True)
        fb_loss.backward()
        self.fb_opt.step()
        # if self.encoder_opt is not None:
        #     self.encoder_opt.step()
        return metrics

    def update_actor(self, obs: torch.Tensor, z: torch.Tensor, old_action: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        perm = torch.randperm(self.cfg.batch_size)
        random_z = z[perm]
        if self.cfg.boltzmann:
            dist = self.actor(obs, z)
            action = dist.rsample()
        else:
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            action = self.actor(obs, z)
            with torch.no_grad():
                random_action = self.actor(obs, random_z)

        # log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        F1, F2 = self.forward_net(obs, z, action)
        Q1 = torch.einsum('sd, sd -> s', F1, z)
        Q2 = torch.einsum('sd, sd -> s', F2, z)
        if self.cfg.additional_metric:
            q1_success = Q1 > Q2
        Q = torch.min(Q1, Q2)
        lmbda = 1/Q.abs().mean().detach()
        actor_loss = (self.cfg.temp * log_prob - lmbda*Q).mean() if self.cfg.boltzmann else -lmbda*Q.mean()
        bc_loss = self.actor.cm.consistency_losses(old_action, obs, z)
        temp_z = torch.zeros_like(z)
        bc_loss1 = self.actor.cm.consistency_losses(random_action, obs, temp_z)
        actor_loss += 0.2 * bc_loss
        actor_loss += 0.2 * bc_loss1

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['bc_loss'] = bc_loss.item()
            metrics['bc_loss1'] = bc_loss1.item()
            metrics['q'] = Q.mean().item()
            if self.cfg.additional_metric:
                metrics['q1_success'] = q1_success.float().mean().item()
            # metrics['actor_logprob'] = log_prob.mean().item()

        return metrics

    def aug_and_encode(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.aug(obs)
        return self.encoder(obs)

    def aug_and_encode_salient(self, obs, z, action):
        obs = self.aug(obs)
        obs_grad = fb_compute_attribution(self.encoder, self.forward_net, obs, z, action.detach())
        mask = compute_attribution_mask(obs_grad,self.quantile)
        masked_obs = obs*mask
        masked_obs[mask<1] = random.uniform(obs.view(-1).min(),obs.view(-1).max())
        return self.encoder(masked_obs)
    
    def update_icm(self, origin_obs, obs, z, action, origin_next_obs, next_obs, step):
        metrics = dict()
        mask_obs = self.aug_and_encode_salient(origin_obs, z, action)
        # with torch.no_grad():
        #     if self.cfg.boltzmann:
        #         dist = self.actor(next_obs, z)
        #         next_action = dist.sample()
        #     else:
        #         stddev = utils.schedule(self.cfg.stddev_schedule, step)
        #         dist = self.actor(next_obs, z, stddev)
        #         next_action = dist.sample(clip=self.cfg.stddev_clip)
        # with torch.no_grad():
        #     mask_next_obs = self.aug_and_encode_salient(origin_next_obs, z, next_action)

        combine_obs = torch.cat([obs, mask_obs], axis=0)
        combine_action = torch.cat([action, action], axis=0)
        combine_next_obs = torch.cat([next_obs, next_obs], axis=0)
        
        forward_error, backward_error = self.icm(combine_obs, combine_action, combine_next_obs)

        loss = forward_error[:self.cfg.batch_size].mean() + backward_error[:self.cfg.batch_size].mean() +\
                0.5*(forward_error[self.cfg.batch_size:].mean() + backward_error[self.cfg.batch_size:].mean())

        self.icm_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        self.icm_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['icm_loss'] = loss.item()

        return metrics
    
    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        if step % self.cfg.update_every_steps != 0:
            return metrics

        batch = replay_loader.sample(self.cfg.batch_size)
        batch = batch.to(self.cfg.device)

        obs = batch.obs
        action = batch.action
        discount = batch.discount
        next_obs = batch.next_obs
        future_obs = batch.future_obs
        
        origin_obs = obs.clone()
        origin_next_obs = next_obs.clone()

        z = self.sample_z(self.cfg.batch_size, device=self.cfg.device)
        if not z.shape[-1] == self.cfg.z_dim:
            raise RuntimeError("There's something wrong with the logic here")

        
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)
            future_obs = self.aug_and_encode(future_obs)
        backward_input = next_obs
        perm = torch.randperm(self.cfg.batch_size)
        backward_input = backward_input[perm]

        if self.cfg.mix_ratio > 0:
            mix_idxs: tp.Any = np.where(np.random.uniform(size=self.cfg.batch_size) < self.cfg.mix_ratio)[0]
            if not self.cfg.rand_weight:
                with torch.no_grad():
                    mix_z = self.backward_net(backward_input[mix_idxs]).detach()
            else:
                # generate random weight
                weight = torch.rand(size=(mix_idxs.shape[0], self.cfg.batch_size)).to(self.cfg.device)
                weight = F.normalize(weight, dim=1)
                uniform_rdv = torch.rand(mix_idxs.shape[0], 1).to(self.cfg.device)
                weight = uniform_rdv * weight
                with torch.no_grad():
                    mix_z = torch.matmul(weight, self.backward_net(backward_input).detach())
            if self.cfg.norm_z:
                mix_z = math.sqrt(self.cfg.z_dim) * F.normalize(mix_z, dim=1)
            z[mix_idxs] = mix_z

        # hindsight replay
        if self.cfg.future_ratio > 0:
            assert future_obs is not None
            future_idxs = np.where(np.random.uniform(size=self.cfg.batch_size) < self.cfg.future_ratio)
            z[future_idxs] = self.backward_net(future_obs[future_idxs]).detach()

        self.save_grid(origin_obs, z, action, step)
        metrics.update(self.update_icm(origin_obs, obs, z, action, origin_next_obs, next_obs, step))
        metrics.update(self.update_fb(obs=obs.detach(), action=action, discount=discount,
                                      next_obs=next_obs.detach(), z=z, step=step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), z, action.detach(), step))

        # update critic target
        utils.soft_update_params(self.forward_net, self.forward_target_net,
                                 self.cfg.fb_target_tau)
        utils.soft_update_params(self.backward_net, self.backward_target_net,
                                 self.cfg.fb_target_tau)

        return metrics

    def save_grid(self, obs, z, action, step):
        temp_obs = obs.clone().detach().to(torch.float32).requires_grad_()
        if step % 10000 == 0:
            self.save_image(temp_obs, z, action, step, prefix="original")

    def save_image(self, obs, z, action, step, prefix="original"):
		# 获取各个梯度图
        grid = make_obs_grid(obs)
        critic_obs_grad = fb_compute_attribution(self.encoder, self.forward_net, obs, z, action.detach())
        critic_grad_grid = make_obs_grad_grid(critic_obs_grad.data.abs())
        # 获取遮挡图
        critic_obs_grad_mask = compute_attribution_mask(critic_obs_grad, quantile=0.9)
        critic_masked_obs = make_obs_grid(obs * critic_obs_grad_mask)	
		# 保存图片
        save_image(grid, os.path.join(self.save_dir, prefix +"_obs_"+str(step)+".jpg"))
        save_image(critic_grad_grid, os.path.join(self.save_dir, prefix +"_critic_grad_"+str(step)+".jpg"))
        save_image(critic_masked_obs, os.path.join(self.save_dir, prefix +"_critic_grad_mask_"+str(step)+".jpg"))