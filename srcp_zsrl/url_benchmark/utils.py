import math
import random
import re
import time
import typing as tp
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
try:
    from typing import Protocol
except ImportError:
    # backward compatible
    from typing_extensions import Protocol  # type: ignore
from captum.attr import GuidedBackprop, GuidedGradCam
from torchvision.utils import make_grid

def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path

class Trainable(Protocol):  # cannot from url_benchmark import agent
    @property
    def training(self) -> bool:
        ...

    def train(self, train: bool) -> None:
        ...


class eval_mode:
    def __init__(self, *models: Trainable) -> None:
        self.models = models
        self.prev_states: tp.List[bool] = []

    def __enter__(self) -> None:
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args: tp.Any) -> None:
        for model, state in zip(self.models, self.prev_states):
            model.train(state)


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


X = tp.TypeVar("X")


def chain(*iterables: tp.Iterable[X]) -> tp.Iterator[X]:
    for it in iterables:
        yield from it


def soft_update_params(net, target_net, tau) -> None:
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def hard_update_params(net, target_net) -> None:
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(param.data)


def to_torch(xs, device) -> tuple:
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m) -> None:
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            # if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            # if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def grad_norm(params, norm_type: float = 2.0):
    params = [p for p in params if p.grad is not None]
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]),
        norm_type)
    return total_norm.item()


def param_norm(params, norm_type: float = 2.0):
    total_norm = torch.norm(
        torch.stack([torch.norm(p.detach(), norm_type) for p in params]),
        norm_type)
    return total_norm.item()


def _repr(obj: tp.Any) -> str:
    items = {x: y for x, y in obj.__dict__.items() if not x.startswith("_")}
    params = ", ".join(f"{x}={y!r}" for x, y in sorted(items.items()))
    return f"{obj.__class__.__name__}({params})"


class Until:
    def __init__(self, until: tp.Optional[int], action_repeat: int = 1) -> None:
        self.until = until
        self.action_repeat = action_repeat

    def __call__(self, step: int) -> bool:
        if self.until is None:
            return True
        until = self.until // self.action_repeat
        return step < until

    def __repr__(self) -> str:
        return _repr(self)


class Every:
    def __init__(self, every: tp.Optional[int], action_repeat: int = 1) -> None:
        self.every = every
        self.action_repeat = action_repeat

    def __call__(self, step: int) -> bool:
        if self.every is None:
            return False
        every = self.every // self.action_repeat
        if step % every == 0:
            return True
        return False

    def __repr__(self) -> str:
        return _repr(self)


class Timer:
    def __init__(self) -> None:
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self) -> tp.Tuple[float, float]:
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self) -> float:
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()) -> torch.Tensor:  # type: ignore
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1) -> None:
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x) -> torch.Tensor:
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x) -> torch.Tensor:
        return x.tanh()

    def _inverse(self, y) -> torch.Tensor:
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y) -> torch.Tensor:
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale) -> None:
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


def schedule(schdl, step) -> float:
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad) -> None:
        super().__init__()
        self.pad = pad

    def forward(self, x) -> torch.Tensor:
        x = x.float()
        n, _, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class PadResizePlus(nn.Module):
    def __init__(self, highest_pad_strength):
        super().__init__()
        self.highest_pad_strength = int(highest_pad_strength)

    def crop(self, imgs, pad_x, pad_y):
        imgs = imgs.to(dtype=torch.float32)
        n, c, h_pad, w_pad = imgs.size()

        # calculate the crop size
        crop_x = w_pad - pad_x
        crop_y = h_pad - pad_y

        # create a grid for cropping
        eps_x = 1.0 / w_pad
        eps_y = 1.0 / h_pad
        
        x_range = torch.linspace(-1.0 + eps_x, 1.0 - eps_x, w_pad, device=imgs.device, dtype=imgs.dtype)[:crop_x]
        y_range = torch.linspace(-1.0 + eps_y, 1.0 - eps_y, h_pad, device=imgs.device, dtype=imgs.dtype)[:crop_y]

        grid_y, grid_x = torch.meshgrid(y_range, x_range)

        base_grid = torch.stack([grid_x, grid_y], dim=-1)
        # print('base_grid.shape', base_grid.shape)

        shift_x = torch.randint(0, pad_x + 1, size=(n, 1, 1, 1), device=imgs.device, dtype=imgs.dtype)
        shift_y = torch.randint(0, pad_y + 1, size=(n, 1, 1, 1), device=imgs.device, dtype=imgs.dtype)
        shift_x *= 2.0 / w_pad
        shift_y *= 2.0 / h_pad
        shift = torch.cat([shift_x, shift_y], dim=-1)
        grid = base_grid + shift
        
        # apply the grid to the input tensor to perform cropping
        padded_imgs_after_crop = F.grid_sample(imgs, grid)

        return padded_imgs_after_crop

    def forward(self, imgs):
        strength = torch.randint(0, self.highest_pad_strength+1, (1,)).item()
        
        _, _, h, w = imgs.shape
        pad_x = torch.randint(0, strength+1, (1,)).item()
        pad_y = strength - pad_x
        # [x+2*pad_x, y+2*pad_y]
        padded_imgs_before_crop = F.pad(imgs, (pad_x, pad_x, pad_y, pad_y))
        # print('padded_imgs_before_crop', padded_imgs_before_crop.shape)
        # [x+pad_x, y+pad_y]
        padded_imgs_after_crop = self.crop(padded_imgs_before_crop, pad_x, pad_y)
        # print('padded_imgs_after_crop', padded_imgs_after_crop.shape)
        # print('######################')

        resize = T.Resize(size=(h, w))

        return resize(padded_imgs_after_crop)
    
class RMS:
    """running mean and std """

    def __init__(self, device, epsilon=1e-4, shape=(1,)) -> None:
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs +
                 torch.square(delta) * self.n * bs /
                 (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S


class PBE:
    """particle-based entropy based on knn normalized by running mean """

    def __init__(self, rms, knn_clip, knn_k, knn_avg, knn_rms, device) -> None:
        self.rms = rms
        self.knn_rms = knn_rms
        self.knn_k = knn_k
        self.knn_avg = knn_avg
        self.knn_clip = knn_clip
        self.device = device

    def __call__(self, rep):
        source = target = rep
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)
        reward, _ = sim_matrix.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        if not self.knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = torch.maximum(
                reward - self.knn_clip,
                torch.zeros_like(reward).to(self.device)
            ) if self.knn_clip >= 0.0 else reward  # (b1, 1)
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = torch.maximum(
                reward - self.knn_clip,
                torch.zeros_like(reward).to(
                    self.device)) if self.knn_clip >= 0.0 else reward
            reward = reward.reshape((b1, self.knn_k))  # (b1, k)
            reward = reward.mean(dim=1, keepdim=True)  # (b1, 1)
        reward = torch.log(reward + 1.0)
        return reward


class FloatStats:

    def __init__(self) -> None:
        self.min = np.inf
        self.max = -np.inf
        self.mean = 0.0
        self.count = 0

    def add(self, value: float) -> "FloatStats":
        self.min = min(value, self.min)
        self.max = max(value, self.max)
        self.count += 1
        self.mean = (self.count - 1) / self.count * self.mean + 1 / self.count * value
        return self

class LinearOutputHook:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
def cal_dormant_ratio(model, *inputs, percentage=0.025):
    hooks = []
    hook_handlers = []
    total_neurons = 0
    dormant_neurons = 0

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = LinearOutputHook()
            hooks.append(hook)
            hook_handlers.append(module.register_forward_hook(hook))

    with torch.no_grad():
        model(*inputs)

    for module, hook in zip(
        (module
         for module in model.modules() if isinstance(module, nn.Linear)),
            hooks):
        with torch.no_grad():
            for output_data in hook.outputs:
                mean_output = output_data.abs().mean(0)
                avg_neuron_output = mean_output.mean()
                dormant_indices = (mean_output < avg_neuron_output *
                                   percentage).nonzero(as_tuple=True)[0]
                total_neurons += module.weight.shape[0]
                dormant_neurons += len(dormant_indices)         

    for hook in hooks:
        hook.outputs.clear()

    for hook_handler in hook_handlers:
        hook_handler.remove()

    return dormant_neurons / total_neurons

class HookFeatures:
    def __init__(self, module):
        self.feature_hook = module.register_forward_hook(self.feature_hook_fn)

    def feature_hook_fn(self, module, input, output):
        self.features = output.clone().detach()
        self.gradient_hook = output.register_hook(self.gradient_hook_fn)

    def gradient_hook_fn(self, grad):
        self.gradients = grad

    def close(self):
        self.feature_hook.remove()
        self.gradient_hook.remove()


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, action=None, trans=False):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.action = action
        self.trans = trans

    def forward(self, obs):
        if self.trans:
            return self.model(obs, self.action)[2]
        if self.action is None:
            return self.model(obs)[0]
        return self.model(obs, self.action)[0]
    
def compute_guided_backprop(obs, action, model, target, trans):
    model = ModelWrapper(model, action=action, trans=trans)
    gbp = GuidedBackprop(model)
    if target == None:
        attribution = gbp.attribute(obs)
    else:
        attribution = gbp.attribute(obs, target = target)
    return attribution

def compute_guided_gradcam(obs, action, model, target, trans):
    obs.requires_grad_()
    obs.retain_grad()
    model = ModelWrapper(model, action=action,  trans=trans)
    gbp = GuidedGradCam(model,layer=model.model.encoder.head_cnn.layers)
    if target == None:
        attribution = attribution = gbp.attribute(obs,attribute_to_layer_input=True)
    else:
        attribution = attribution = gbp.attribute(obs,attribute_to_layer_input=True, target = target)
    return attribution

def compute_vanilla_grad(critic_target, obs, action):
    obs.requires_grad_()
    obs.retain_grad()
    q, q2 = critic_target(obs, action.detach())
    q.sum().backward()
    return obs.grad


def compute_attribution(model, obs, action=None,method="guided_backprop", target = None, trans=False):
    if method == "guided_backprop":
        return compute_guided_backprop(obs, action, model, target, trans)
    if method == 'guided_gradcam':
        return compute_guided_gradcam(obs,action,model, target, trans)
    return compute_vanilla_grad(model, obs, action)


def compute_features_attribution(critic_target, obs, action):
    obs.requires_grad_()
    obs.retain_grad()
    hook = HookFeatures(critic_target.encoder)
    q, _ = critic_target(obs, action.detach())
    q.sum().backward()
    features_gardients = hook.gradients
    hook.close()
    return obs.grad, features_gardients


def compute_attribution_mask(obs_grad, quantile=0.95):
    mask = []
    for i in [0, 3, 6]:
        attributions = obs_grad[:, i : i + 3].abs().max(dim=1)[0]
        q = torch.quantile(attributions.flatten(1), quantile, 1)
        mask.append((attributions >= q[:, None, None]).unsqueeze(1).repeat(1, 3, 1, 1))
    return torch.cat(mask, dim=1)

def my_compute_attribution_mask(obs_grad, quantile=0.95):
    mask = []
    for i in [0, 3, 6]:
        attributions = obs_grad[:, i : i + 3].abs().max(dim=1)[0]
        # 获取尺寸
        n, h, w = attributions.size()
        flatten_attributions = attributions.flatten(1)
        max_values, _ = torch.max(flatten_attributions, dim=1, keepdim=True)
        is_zero_max = max_values == 0
        max_values = max_values.masked_fill(is_zero_max, 1)
        normalized_attributions = flatten_attributions / max_values
        temp_mask = normalized_attributions.reshape((n,h,w))
        mask.append(temp_mask.unsqueeze(1).repeat(1, 3, 1, 1))
    return torch.cat(mask, dim=1)

def make_obs_grid(obs, n=4):
    sample = []
    for i in range(n):
        for j in range(0, 9, 3):
            sample.append(obs[i, j : j + 3].unsqueeze(0))
    sample = torch.cat(sample, 0)
    return make_grid(sample, nrow=3) / 255.0


def make_attribution_pred_grid(attribution_pred, n=4):
    return make_grid(attribution_pred[:n], nrow=1)


def make_obs_grad_grid(obs_grad, n=4):
    sample = []
    for i in range(n):
        for j in range(0, 9, 3):
            channel_attribution, _ = torch.max(obs_grad[i, j : j + 3], dim=0)
            sample.append(channel_attribution[(None,) * 2] / channel_attribution.max())
    sample = torch.cat(sample, 0)
    q = torch.quantile(sample.flatten(1), 0.97, 1)
    sample[sample <= q[:, None, None, None]] = 0
    return make_grid(sample, nrow=3)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, action=None, trans=False):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.action = action
        self.trans = trans

    def forward(self, obs):
        if self.trans:
            return self.model(obs, self.action)[2]
        if self.action is None:
            return self.model(obs)[0]
        return self.model(obs, self.action)[0]
    
class FBModelWrapper(torch.nn.Module):
    def __init__(self, encoder, model, z=None, action=None):
        super(FBModelWrapper, self).__init__()
        self.encoder = encoder
        self.model = model
        self.z = z
        self.action = action

    def forward(self, obs):
        # print("obs1.requires_grad", obs.requires_grad)
        obs = self.encoder(obs)
        # print("obs2.requires_grad", obs.requires_grad)
        F1, F2 = self.model(obs, self.z, self.action)
        Q = torch.einsum('sd, sd -> s', F1, self.z)
        return Q
    
def fb_compute_guided_backprop(obs, z, action, encoder, model, target):
    # print("obs.requires_grad, z.requires_grad, action.requires_grad", obs.requires_grad, z.requires_grad, action.requires_grad)
    model = FBModelWrapper(encoder, model, z=z, action=action)
    gbp = GuidedBackprop(model)
    if target == None:
        attribution = gbp.attribute(obs)
    else:
        attribution = gbp.attribute(obs, target = target)
    return attribution

def fb_compute_guided_gradcam(obs, z, action, encoder, model, target):
    obs.requires_grad_()
    obs.retain_grad()
    model = FBModelWrapper(encoder, model, action=action)
    gbp = GuidedGradCam(model,layer=model.model.encoder.head_cnn.layers)
    if target == None:
        attribution = attribution = gbp.attribute(obs,attribute_to_layer_input=True)
    else:
        attribution = attribution = gbp.attribute(obs,attribute_to_layer_input=True, target = target)
    return attribution

def fb_compute_vanilla_grad(encoder, model, z, obs, action):
    obs.requires_grad_()
    obs.retain_grad()
    obs = encoder(obs)
    F1, F2 = model(obs, z.detach(), action.detach())
    Q = torch.einsum('sd, sd -> s', F1, z)
    Q.sum().backward()
    return obs.grad

def fb_compute_attribution(encoder, model, obs, z=None, action=None,method="guided_backprop", target = None):
    if method == "guided_backprop":
        return fb_compute_guided_backprop(obs, z, action, encoder, model, target)
    if method == 'guided_gradcam':
        return fb_compute_guided_gradcam(obs, z, action, encoder, model, target)
    return fb_compute_vanilla_grad(encoder, model, z, obs, action)


class FBModelWrapper1(torch.nn.Module):
    def __init__(self, encoder, model, z=None, action=None):
        super(FBModelWrapper1, self).__init__()
        self.encoder = encoder
        self.model = model
        self.z = z
        self.action = action

    def forward(self, obs):
        # print("obs1.requires_grad", obs.requires_grad)
        obs = self.encoder(obs, record=True)
        # print("obs2.requires_grad", obs.requires_grad)
        F1, F2 = self.model(obs, self.z, self.action)
        Q = torch.einsum('sd, sd -> s', F1, self.z)
        return Q
    
def fb_compute_guided_backprop1(obs, z, action, encoder, model, target):
    # print("obs.requires_grad, z.requires_grad, action.requires_grad", obs.requires_grad, z.requires_grad, action.requires_grad)
    model = FBModelWrapper1(encoder, model, z=z, action=action)
    gbp = GuidedBackprop(model)
    if target == None:
        attribution = gbp.attribute(obs)
    else:
        attribution = gbp.attribute(obs, target = target)
    return attribution

def fb_compute_guided_gradcam1(obs, z, action, encoder, model, target):
    obs.requires_grad_()
    obs.retain_grad()
    model = FBModelWrapper1(encoder, model, action=action)
    gbp = GuidedGradCam(model,layer=model.model.encoder.head_cnn.layers)
    if target == None:
        attribution = attribution = gbp.attribute(obs,attribute_to_layer_input=True)
    else:
        attribution = attribution = gbp.attribute(obs,attribute_to_layer_input=True, target = target)
    return attribution

def fb_compute_vanilla_grad1(encoder, model, z, obs, action):
    obs.requires_grad_()
    obs.retain_grad()
    obs = encoder(obs, record=True)
    F1, F2 = model(obs, z.detach(), action.detach())
    Q = torch.einsum('sd, sd -> s', F1, z)
    Q.sum().backward()
    return obs.grad

def fb_compute_attribution1(encoder, model, obs, z=None, action=None,method="guided_backprop", target = None):
    if method == "guided_backprop":
        return fb_compute_guided_backprop1(obs, z, action, encoder, model, target)
    if method == 'guided_gradcam':
        return fb_compute_guided_gradcam1(obs, z, action, encoder, model, target)
    return fb_compute_vanilla_grad1(encoder, model, z, obs, action)

class FBModelWrapper2(torch.nn.Module):
    def __init__(self, encoder, model, z=None, action=None, act_tok=None):
        super(FBModelWrapper2, self).__init__()
        self.encoder = encoder
        self.model = model
        self.z = z
        self.action = action
        self.act_tok=act_tok

    def forward(self, obs):
        # print("obs1.requires_grad", obs.requires_grad)
        obs = self.encoder(obs)
        # print("obs2.requires_grad", obs.requires_grad)
        F1, F2 = self.model(obs, self.z, self.action, self.act_tok)
        Q = torch.einsum('sd, sd -> s', F1, self.z)
        return Q
    
def fb_compute_guided_backprop2(obs, z, action, encoder, model, target, act_tok):
    # print("obs.requires_grad, z.requires_grad, action.requires_grad", obs.requires_grad, z.requires_grad, action.requires_grad)
    model = FBModelWrapper2(encoder, model, z=z, action=action, act_tok=act_tok)
    gbp = GuidedBackprop(model)
    if target == None:
        attribution = gbp.attribute(obs)
    else:
        attribution = gbp.attribute(obs, target = target)
    return attribution

def fb_compute_guided_gradcam2(obs, z, action, encoder, model, target, act_tok):
    obs.requires_grad_()
    obs.retain_grad()
    model = FBModelWrapper2(encoder, model, action=action, act_tok=act_tok)
    gbp = GuidedGradCam(model,layer=model.model.encoder.head_cnn.layers)
    if target == None:
        attribution = attribution = gbp.attribute(obs,attribute_to_layer_input=True)
    else:
        attribution = attribution = gbp.attribute(obs,attribute_to_layer_input=True, target = target)
    return attribution

def fb_compute_vanilla_grad2(encoder, model, z, obs, action, act_tok):
    obs.requires_grad_()
    obs.retain_grad()
    obs = encoder(obs)
    F1, F2 = model(obs, z.detach(), action.detach(), act_tok)
    Q = torch.einsum('sd, sd -> s', F1, z)
    Q.sum().backward()
    return obs.grad

def fb_compute_attribution2(encoder, model, obs, z=None, action=None, act_tok=None, method="guided_backprop", target = None):
    if method == "guided_backprop":
        return fb_compute_guided_backprop2(obs, z, action, encoder, model, target, act_tok)
    if method == 'guided_gradcam':
        return fb_compute_guided_gradcam2(obs, z, action, encoder, model, target, act_tok)
    return fb_compute_vanilla_grad2(encoder, model, z, obs, action, act_tok)


class ICMModelWrapper(torch.nn.Module):
    def __init__(self, encoder, model, action=None):
        super(ICMModelWrapper, self).__init__()
        self.encoder = encoder
        self.model = model
        self.action = action

    def forward(self, obs):
        # print("obs1.requires_grad", obs.requires_grad)
        obs = self.encoder(obs)
        # print("obs2.requires_grad", obs.requires_grad)
        next_obs = self.model.forward_once(obs, self.action)
        # print("next_obs", next_obs.shape)
        return torch.sum(next_obs, dim=-1)
    
def icm_compute_guided_backprop(obs, action, encoder, model, target):
    # print("obs.requires_grad, z.requires_grad, action.requires_grad", obs.requires_grad, z.requires_grad, action.requires_grad)
    model = ICMModelWrapper(encoder, model, action=action)
    gbp = GuidedBackprop(model)
    if target == None:
        attribution = gbp.attribute(obs)
    else:
        attribution = gbp.attribute(obs, target = target)
    return attribution

def icm_compute_guided_gradcam(obs, action, encoder, model, target):
    obs.requires_grad_()
    obs.retain_grad()
    model = ICMModelWrapper(encoder, model, action=action)
    gbp = GuidedGradCam(model,layer=model.model.encoder.head_cnn.layers)
    if target == None:
        attribution = attribution = gbp.attribute(obs,attribute_to_layer_input=True)
    else:
        attribution = attribution = gbp.attribute(obs,attribute_to_layer_input=True, target = target)
    return attribution

def icm_compute_vanilla_grad(encoder, model, obs, action):
    obs.requires_grad_()
    obs.retain_grad()
    obs = encoder(obs, record=True)
    Q1, Q2 = model(obs, action.detach())
    Q1.sum().backward()
    return obs.grad



def icm_compute_attribution(encoder, model, obs, action=None, method="guided_backprop", target = None):
    if method == "guided_backprop":
        return icm_compute_guided_backprop(obs, action, encoder, model, target)
    if method == 'guided_gradcam':
        return icm_compute_guided_gradcam(obs, action, encoder, model, target)
    return icm_compute_vanilla_grad(encoder, model, obs, action)


class ActionEncoding(nn.Module):
    def __init__(self, action_dim, latent_action_dim, multistep):
        super().__init__()
        self.action_dim = action_dim
        self.action_tokenizer = nn.Sequential(
            nn.Linear(action_dim, 64), nn.Tanh(),
            nn.Linear(64, latent_action_dim)
        )
        self.action_seq_tokenizer = nn.Sequential(
            nn.Linear(latent_action_dim*multistep, latent_action_dim*multistep),
            nn.LayerNorm(latent_action_dim*multistep), nn.Tanh()
        )
        self.apply(weight_init)
        
    def forward(self, action, seq=False):
        if seq:
            batch_size = action.shape[0]
            action = self.action_tokenizer(action) #(batch_size, length_action_dim)
            action = action.reshape(batch_size, -1)
            return self.action_seq_tokenizer(action)
        else:
            return self.action_tokenizer(action)