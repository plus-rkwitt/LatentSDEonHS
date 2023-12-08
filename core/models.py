"""Implementation of model parts used throughout all experiments."""

import math
import numpy as np
from typing import Tuple, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import reduce

from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.distributions.normal import Normal
from torch.distributions.continuous_bernoulli import ContinuousBernoulli
from torch.distributions import (
    Distribution,
    kl_divergence,
    Bernoulli,
    OneHotCategorical,
)

from core.power_spherical.distributions import PowerSpherical, HypersphericalUniform
from core.pathdistribution import SOnPathDistribution, BrownianMotionOnSphere, PathDistribution
from utils.misc import scatter_obs_and_msk


class MultiTimeAttention(nn.Module):
    """source: https://github.com/reml-lab/mTAN/blob/main/src/models.py (modified)

    Args:
        input_dim (int): _description_
        nhidden (int, optional): _description_. Defaults to "16".
        embed_time (int, optional): _description_. Defaults to "16".
        num_heads (int, optional): _description_. Defaults to "1".
    """

    def __init__(
        self,
        input_dim: int,
        nhidden: int = 16,
        embed_time: int = 16,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        assert embed_time % num_heads == 0
        self.input_dim = input_dim
        self.embed_time = embed_time
        self.num_heads = num_heads
        self.embed_time_k = embed_time // num_heads
        self.nhidden = nhidden
        self.linears = nn.ModuleList(
            [
                nn.Linear(embed_time, embed_time),
                nn.Linear(embed_time, embed_time),
                nn.Linear(input_dim * num_heads, nhidden),
            ]
        )

    def extra_repr(self) -> str:
        return (
            "input_dim={self.input_dim}, nhidden={self.nhidden}, "
            f"embed_time={self.embed_time}, num_heads={self.num_heads})"
        )        

    def attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        dropout: Optional[nn.Dropout] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute 'Scaled Dot Product Attention'

        Args:
            query (Tensor): _description_
            key (Tensor): _description_
            value (Tensor): _description_
            mask (Tensor, optional): _description_. Defaults to None.
            dropout (nn.Dropout, optional): _description_. Defaults to None.

        Returns:
            (Tuple[Tensor, Tensor]): _description_
        """
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn * value.unsqueeze(-3), -2), p_attn

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        dropout: Optional[nn.Dropout] = None,
    ) -> Tensor:
        """Compute 'Scaled Dot Product Attention'

        Args:
            query (Tensor): _description_
            key (Tensor): _description_
            value (Tensor): _description_
            mask (Tensor, optional): _description_. Defaults to None.
            dropout (nn.Dropout, optional): _description_. Defaults to None.

        Returns:
            (Tensor): _description_
        """
        batch, _, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [
            l(x).view(x.size(0), -1, self.num_heads, self.embed_time_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key))
        ]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.num_heads * dim)
        return self.linears[-1](x)


class EncMtanRnn(nn.Module):
    """source: https://github.com/reml-lab/mTAN/blob/main/src/models.py (modified, to allow setting the device with .to(device) )

    Args:
        input_dim (int): _description_
        query (Tensor): _description_
        latent_dim (int, optional): _description_. Defaults to "2".
        nhidden (int, optional): _description_. Defaults to "16".
        embed_time (int, optional): _description_. Defaults to "16".
        num_heads (int, optional): _description_. Defaults to "1".
        learn_emb (bool, optional): _description_. Defaults to "False".
    """

    def __init__(
        self,
        input_dim: int,
        query: Tensor,
        latent_dim: int = 2,
        nhidden: int = 16,
        embed_time: int = 16,
        num_heads: int = 1,
        learn_emb: bool = False
    ) -> None:
        super().__init__()
        self.input_dim = input_dim  # self.dim
        self.register_buffer('query', query)
        self.latent_dim = latent_dim
        self.nhidden = nhidden
        self.embed_time = embed_time
        self.num_heads = num_heads
        self.learn_emb = learn_emb
        self.att = MultiTimeAttention(2 * input_dim, nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2 * nhidden, 50), nn.ReLU(), nn.Linear(50, latent_dim * 2)
        )
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time - 1)
            self.linear = nn.Linear(1, 1)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, query=Tensor: {self.query.shape}, "
            f"latent_dim={self.latent_dim}, nhidden={self.nhidden}, "
            f"embed_time={self.embed_time}, num_heads={self.num_heads}, "
            f"learn_emb={self.learn_emb})"
        )

    def learn_time_embedding(self, tt: Tensor) -> Tensor:
        """_summary_

        Args:
            tt (Tensor): _description_

        Returns:
            (Tensor): _description_
        """
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def fixed_time_embedding(self, pos: Tensor) -> Tensor:
        """_summary_

        Args:
            pos (Tensor): _description_

        Returns:
            (Tensor): _description_
        """
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.0 * pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: Tensor, time_steps: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): _description_
            time_steps (Tensor): _description_

        Returns:
            (Tensor): _description_
        """
        time_steps = time_steps
        mask = x[:, :, self.dim :]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
            query = self.learn_time_embedding(self.query.unsqueeze(0))
        else:
            key = self.fixed_time_embedding(time_steps)
            query = self.fixed_time_embedding(self.query.unsqueeze(0))
        out = self.att(query, key, x, mask)
        out, _ = self.gru_rnn(out)
        out = self.hiddens_to_z0(out)
        return out


class MTANEncoder(EncMtanRnn):
    """_summary_

    Args:
        input_dim (int): _description_
        query (Tensor): _description_
        latent_dim (int, optional): _description_. Defaults to "2".
        nhidden (int, optional): _description_. Defaults to "16".
        embed_time (int, optional): _description_. Defaults to "16".
        num_heads (int, optional): _description_. Defaults to "1".
        learn_emb (bool, optional): _description_. Defaults to "False".
    """

    def __init__(
        self,
        input_dim: int,
        query: Tensor,
        latent_dim: int = 2,
        nhidden: int = 16,
        embed_time: int = 16,
        num_heads: int = 1,
        learn_emb: bool = False
    ) -> None:
        super().__init__(
            input_dim,
            query,
            latent_dim,
            nhidden,
            embed_time,
            num_heads,
            learn_emb
        )
        self.input_dim = input_dim
        self.query = query
        self.latent_dim = latent_dim
        self.nhidden = nhidden
        self.embed_time = embed_time
        self.num_heads = num_heads
        self.learn_emb = learn_emb
        self.hiddens_to_z0 = None
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=False, batch_first=True)

    def __repr__(self) -> str:
        return (
            f"MTANEncoder(input_dim={self.input_dim}, query={self.query}, "
            f"latent_dim={self.latent_dim}, nhidden={self.nhidden}, "
            f"embed_time={self.embed_time}, num_heads={self.num_heads}, "
            f"learn_emb={self.learn_emb})"
        )

    def forward(self, x: Tensor, time_steps: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): _description_
            time_steps (Tensor): _description_

        Returns:
            (Tensor): _description_
        """
        mask = x[:, :, self.input_dim :]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
            query = self.learn_time_embedding(self.query.unsqueeze(0))
        else:
            key = self.fixed_time_embedding(time_steps)
            query = self.fixed_time_embedding(self.query.unsqueeze(0))
        out = self.att(query, key, x, mask)
        _, out = self.gru_rnn(out)
        return out.squeeze(0)


class Chebyshev(nn.Module):
    r"""Chebyshev is a class that parameterizes the drift component :math:``K^{\phi}``;
        cf. Sec. 3.4.

    Args:
        input_dim (int): Number of channels of the input
        degree (int): Degree of the individual polynomial
        out_dim (int): Number of channels of the output
        time_min (float, optional): The integer which defines the left function domain bound.
                                    Defaults to "0".
        time_max (float, optional): The integer which defines the right function domain bound.
                                    Defaults to "8".
    """

    def __init__(
        self,
        input_dim: int,
        degree: int,
        out_dim: int,
        time_min: float = 0.,
        time_max: float = 1.,
    ) -> None:
        super().__init__()
        self.degree = degree
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.time_min = time_min
        self.time_max = time_max
        self.map = nn.Linear(input_dim, degree * out_dim)
        self.register_buffer("degrees", torch.arange(degree))

    def __repr__(self) -> str:
        return (
            f"""Chebyshev(input_dim={self.input_dim}, degree={self.degree}, """
            f"""out_dim={self.out_dim}, time_min={self.time_min}, time_max={self.time_max})"""
        )

    def interval_transform(self, t: float) -> float:
        r"""Chebyshev polynomials (of the first kind) form a complete orthogonal basis for
            functions on "[-1,1]". This is a interval transform for using arbitrary intervals
            "[a,b]".

        Args:
            t (float): time, greater than "a" and smaller than "b"

        Returns:
            (float): time, greater than "-1" and smaller than "1"
        """
        # TODO: Should possibly be done as interval_transform = lambda t : (2*t - (self.time_min + self.time_max)) / (self.time_max - self.time_min)
        return (t - self.time_min) / (self.time_max - self.time_min)

    def forward(self, out_encoder: Tensor, time_steps: Tensor) -> Tensor:
        r"""
        Args:
            out_encoder (Tensor): tensor of shape "(N: batchsize, d: \# of channels)"
            time_steps (Tensor): tensor of shape "(L: \# of time steps)"

        Returns:
            (Tensor): tensor of shape "(N: batchsize, L: \# of time steps,
                                 d: \# of channels)"
        """
        time_steps = self.interval_transform(time_steps)
        monomials = torch.cos(torch.outer(torch.arccos(time_steps), self.degrees))
        coefficients = self.map(out_encoder).view(
            out_encoder.shape[0], self.out_dim, self.degree
        )
        polynomial = F.linear(coefficients, monomials)
        polynomial = polynomial.permute(0, 2, 1)
        return polynomial


class UnFlatten(nn.Module):
    """UnFlatten is a class to flatten a tensor dimension, expanding it to a desired shape.

    Args:
        w (int): _description_
    """

    def __init__(self, w: int) -> None:
        super().__init__()
        self.w = w

    def __repr__(self) -> str:
        return f"UnFlatten(w={self.w})"

    def forward(self, inp: Tensor) -> Tensor:
        r"""
        Args:
            inp (Tensor): input tensor of shape "(N: batchsize, L: \# of time steps,
                            d: \# of channels)"

        Returns:
            (Tensor): tensor of shape "(N: batchsize, TODO,
                             d1: w, d2: w)"
        """
        nc = inp[0].numel() // (self.w**2)  # TODO: what's nc?
        return inp.view(inp.size(0), nc, self.w, self.w)


class RotatingMNISTRecogNetwork(nn.Module):
    """_summary_

    Args:
        n_filters (int, optional): _description. Defaults to "8".
    """

    def __init__(self, n_filters: int = 8) -> None:
        super().__init__()
        self.n_filters = n_filters
        self.map = nn.Sequential(
            # 28x28
            nn.Conv2d(1, n_filters, kernel_size=5, stride=2, padding=(2, 2)),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            # 14x14
            nn.Conv2d(
                n_filters, n_filters * 2, kernel_size=5, stride=2, padding=(2, 2)
            ),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(),
            # 7x7
            nn.Conv2d(
                n_filters * 2, n_filters * 4, kernel_size=5, stride=2, padding=(2, 2)
            ),
            nn.BatchNorm2d(n_filters * 4),
            nn.ReLU(),
            # 4x4
            nn.Conv2d(
                n_filters * 4, n_filters * 8, kernel_size=5, stride=2, padding=(2, 2)
            ),
            nn.BatchNorm2d(n_filters * 8),
            nn.ReLU(),
            nn.Flatten(),
        )

    def extra_repr(self):
        return f"n_filters  = {self.n_filters}"


    def forward(self, inp: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """_summary_

        Args:
            inp (Tuple[Tensor, Tensor, Tensor]): inp comes in as generic 3-tuple of (obs, obs_msk, obs_tps)

        Returns:
            (Tensor): _description_
        """

        h = self.map(inp[0])
        return h


class RotatingMNISTReconNetwork(nn.Module):
    """_summary_

    Args:
        z_dim (int): _description_
        n_filters (int, optional): _description_. Defaults to "16".
    """

    def __init__(self, z_dim: int, n_filters: int = 16) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.n_filters = n_filters
        self.map = nn.Sequential(
            nn.Linear(z_dim, 3 * 3 * 8),
            UnFlatten(3),
            nn.ConvTranspose2d(
                8, n_filters * 8, kernel_size=3, stride=2, padding=(0, 0)
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                n_filters * 8,
                n_filters * 4,
                kernel_size=5,
                stride=2,
                padding=(2, 2),
                output_padding=(1, 1),
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                n_filters * 4,
                n_filters * 2,
                kernel_size=5,
                stride=2,
                padding=(2, 2),
                output_padding=(1, 1),
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                n_filters * 2, 1, kernel_size=5, stride=1, padding=(2, 2)
            ),
        )

    def extra_repr(self) -> str:
        return (
            f"z_dim={self.z_dim}, n_filters={self.n_filters}"
        )

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): _description_

        Returns:
            (Tensor): _description_
        """
        return self.map(x)


class SOnPathDistributionEncoder(nn.Module):
    """_summary_

    Args:
        loc_map (nn.Module): _description_
        scl_map (nn.Module): _description_
        time_fn (nn.Module): _description_
        learnable_prior (bool, optional): Defaults to False.
        in_dim (int, optional): Defaults to "32".
    """

    def __init__(
        self,
        loc_map: nn.Module,
        scl_map: nn.Module,
        time_fn: nn.Module,
        learnable_prior: bool = False,
        in_dim: int = 32,
    ) -> None:
        super().__init__()
        self._loc_map = loc_map
        self._scl_map = scl_map
        self._time_fn = time_fn
        self._sigma = nn.Parameter(torch.tensor(0.1))
        self._learnable_prior = learnable_prior
        self.in_dim = in_dim
        if learnable_prior:
            self.prior_h = nn.Parameter(torch.randn(1, in_dim))

    def extra_repr(self) -> str:
        return f"learnable prior={self._learnable_prior}"
        

    def forward(self, h: Tensor, t: Tensor) -> Tuple[PathDistribution, PathDistribution]:
        """_summary_

        Args:
            h (Tensor): _description_
            t (Tensor): _description_

        Returns:
            (Tuple[PathDistribution, PathDistribution]): _description_
        """
        loc = self._loc_map(h)
        scl = self._scl_map(h)        
        scl = scl.square() * 100.0
        scl = torch.minimum(scl, torch.tensor(50000.0, device=scl.device))
        p0 = PowerSpherical(F.normalize(loc), scl.squeeze(dim=1))

        def K(arg_t: Tensor) -> Tensor: return self._time_fn(h, arg_t)
        posterior = SOnPathDistribution(p0, K, self._sigma, t)

        if self._learnable_prior:
            prior_p0 = HypersphericalUniform(loc.shape[1], h.device, h.dtype)
            def prior_K(arg_t: Tensor) -> Tensor: 
                return self._time_fn(self.prior_h, arg_t)
            prior = SOnPathDistribution(prior_p0, prior_K, self._sigma, t)
        else:
            prior = BrownianMotionOnSphere(loc.shape[-1], self._sigma, t)
        return posterior, prior


class PendulumRecogNetwork(nn.Module):
    """_summary_

    Args:
        h_dim (int, optional): _description_. Defaults to "32".
        mtan_input_dim (int, optional): _description Defaults to "32".
    """

    def __init__(
        self, 
        h_dim: int = 32, 
        mtan_input_dim: int = 32,
        use_atanh: bool = False
    ) -> None:
        super().__init__()
        learn_emb = True  # we always want to learn a time embedding
        self.use_atanh = use_atanh
        self.h_dim = h_dim
        self.mtan_input_dim = mtan_input_dim
        self.mtan = MTANEncoder(
            input_dim = mtan_input_dim,
            query=torch.linspace(0, 1.0, 128),
            nhidden = h_dim,
            embed_time=128,
            num_heads=1,
            learn_emb = learn_emb
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=12, out_channels=12, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=108, out_features=mtan_input_dim),
            nn.Tanh(),
        )


    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): _description_

        Returns:
            (Tensor): _description_
        """
        observed_data, observed_mask, observed_tp = x
        batch_size, time_points, _, _, _ = observed_data.shape
        observed_data = observed_data.flatten(0, 1)
        h = self.cnn(observed_data)
        h = h.unflatten(0, (batch_size, time_points))
        # uses that mask[i,j,:,:,:] is either const 0 or 1
        observed_mask = observed_mask[:, :, 0:1, 0, 0].expand_as(h)
        h = self.mtan(torch.cat((h, observed_mask), 2), observed_tp)
        if self.use_atanh:
            eps = 1e-5
            h = h - h.sign() * eps
            h = h.atanh()
        return h


class RKNLayerNorm(nn.Module):
    """source: https://github.com/ALRhub/rkn_share/ (modified)

    Args:
        channels (int): _description_
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self._scale = torch.nn.Parameter(torch.ones(1, self.channels, 1, 1))
        self._offset = torch.nn.Parameter(torch.zeros(1, self.channels, 1, 1))

    def extra_repr(self) -> str:
        return f"channels={self.channels}"

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): _description_

        Returns:
            (Tensor): _description_
        """
        normalized = (x - x.mean(dim=[-3, -2, -1], keepdim=True)) / x.std(
            dim=[-3, -2, -1], keepdim=True
        )
        return self._scale * normalized + self._offset


class PendulumReconNetwork(nn.Module):
    """_summary_

    Args:
        z_dim (int, optional): _description_. Defaults to "16".
    """

    def __init__(self, z_dim: int = 16) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.map = nn.Sequential(
            nn.Linear(in_features = z_dim, out_features=144),
            nn.ReLU(),
            nn.Unflatten(1, [16, 3, 3]),
            nn.ConvTranspose2d(
                in_channels=16, out_channels=16, kernel_size=5, stride=4, padding=2
            ),
            RKNLayerNorm(channels=16),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=16, out_channels=12, kernel_size=3, stride=2, padding=1
            ),
            RKNLayerNorm(channels=12),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=12, out_channels=1, kernel_size=2, stride=2, padding=5
            ),
            nn.Sigmoid(),
        )

    def extra_repr(self) -> str:
        return f"z_dim={self.z_dim}"

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): _description_

        Returns:
            (Tensor): _description_
        """
        return self.map(x)


class PathToGaussianDecoder(nn.Module):
    """_summary_

    Args:
        mu_map (nn.Module): _description_
        sigma_map (nn.Module, optional): _description_. Defaults to "None".
        initial_sigma (float, optional): _description_. Defaults to "1".
    """

    def __init__(self, 
                mu_map: nn.Module,
                sigma_map: Optional[nn.Module] = None,
                initial_sigma: float = 1.
    ) -> None:
        super().__init__()
        self.mu_map = mu_map
        self.sigma_map = sigma_map
        self.initial_sigma = initial_sigma
        if self.sigma_map is None:
            self.sigma = nn.Parameter(torch.tensor(initial_sigma))

    def extra_repr(self) -> str:
        if self.sigma_map is None:
            s = f"initial_sigma={self.initial_sigma}"
        else:
            s=""
        return s

    def forward(self, x: Tensor) -> Normal:
        """_summary_

        Args:
            x (Tensor): _description_

        Returns:
            (Normal): _description_
        """
        n_samples, batch_size, time_steps, _ = x.shape
        target_shape = [1 for i in range(len(x.shape))]

        mu = self.mu_map(x.flatten(0, 2))
        mu = mu.unflatten(0, (n_samples, batch_size, time_steps))
        
        if self.sigma_map is not None:
            sigma = self.sigma_map(x)
        else:
            sigma = self.sigma.view(target_shape).expand_as(mu)            
        return Normal(mu, sigma.square())


class PathToContinuousBernoulliDecoder(nn.Module):
    """_summary_

    Args:
        logit_map (nn.Module): _description_
    """

    def __init__(self, logit_map: nn.Module) -> None:
        super().__init__()
        self.logit_map = logit_map

    def forward(self, x: Tensor) -> ContinuousBernoulli:
        """_summary_

        Args:
            x (Tensor): _description_

        Returns:
            (ContinuousBernoulli): _description_
        """
        n_samples, batch_size, time_steps, _ = x.shape

        logits = self.logit_map(x.flatten(0, 2))
        logits = logits.unflatten(0, (n_samples, batch_size, time_steps))
        return ContinuousBernoulli(logits=logits)


class PathToBernoulliDecoder(nn.Module):
    """_summary_

    Args:
        logit_map (nn.Module): _description_
    """

    def __init__(self, logit_map: nn.Module) -> None:
        super().__init__()
        self.logit_map = logit_map

    def forward(self, x: Tensor) -> Bernoulli:
        """_summary_

        Args:
            x (Tensor): _description_

        Returns:
            (Bernoulli): _description_
        """
        n_samples, batch_size, time_steps, _ = x.shape

        logits = self.logit_map(x.flatten(0, 2))
        logits = logits.unflatten(0, (n_samples, batch_size, time_steps))
        p = Bernoulli(logits=logits)
        p.support = torch.distributions.constraints.interval(0, 1)
        return p


class PathToCategoricalDecoder(nn.Module):
    """_summary_

    Args:
        logit_map (nn.Module): _description_
    """

    def __init__(self, logit_map: nn.Module) -> None:
        super().__init__()
        self.logit_map = logit_map

    def forward(self, x: Tensor) -> OneHotCategorical:
        """_summary_

        Args:
            x (Tensor): _description_

        Returns:
            OneHotCategorical: _description_
        """
        n_samples, batch_size, time_steps, _ = x.shape

        logits = self.logit_map(x.flatten(0, 2))
        logits = logits.unflatten(0, (n_samples, batch_size, time_steps))
        p = OneHotCategorical(logits=logits)
        return p


class PerTimePointCrossEntropyLoss(_Loss):
    """_summary_

    Args:
        reduction (str): _description_. Defaults to "none".
    """

    def __init__(self, reduction: str = "none") -> None:
        super().__init__(reduction = reduction)
        self.reduction = reduction

    def __repr__(self) -> str:
        return f"PerTimePointCrossEntropyLoss(reduction={self.reduction})"

    def forward(self, inp: Tensor, target: Tensor, tps: Tensor) -> Tensor:
        """_summary_

        Args:
            inp (Tensor): _description_
            target (Tensor): _description_
            tps (Tensor): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            (Tensor): _description_
        """
        mc_samples, input_batch_len, _, dim = inp.shape
        target_batch_len, target_num_time_points = target.shape

        assert input_batch_len == target_batch_len
        batch_len = input_batch_len

        idx = (
            tps.view([1, batch_len, target_num_time_points, 1])
            .repeat(mc_samples, 1, 1, dim)
            .long()
        )
        input_at_tps = inp.gather(2, idx).mean(
            dim=0
        )  # -> [batch_len, aux_num_time_points, dim (num_classes)]
        loss = torch.nn.functional.cross_entropy(
            input_at_tps.flatten(0, 1), target.flatten(), reduction="none"
        )
        loss = loss.view(batch_len, target_num_time_points).mean(dim=1, keepdim=True)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError()


class SinCosLoss(_Loss):
    """Loss on (sin,cos) tuples.

    Args:
        reduction (str): 'mean', 'sum', or 'none'
            'none' returns a per-batch-element loss, whereas
            'mean' and 'sum' perform aggregation over the
            per-batch-element loss values. Defaults to "none".
    """

    def __init__(self, reduction: str = "none") -> None:
        super().__init__(reduction = reduction)
        self.reduction = reduction

    def __repr__(self) -> str:
        return f"SinCosLoss(reduction={self.reduction})"

    def forward(self, inp: Tensor, target: Tensor, tps: Tensor) -> Tensor:
        """_summary_

        Args:
            inp (Tensor): _description_
            target (Tensor): _description_
            tps (Tensor): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            (Tensor): _description_
        """
        idx = tps.long().unsqueeze(-1).repeat(1, 1, 1, 2)
        val_sincos = inp.gather(2, idx)
        tgt_sincos = target.unsqueeze(0).expand(val_sincos.shape[0:1] + target.shape)
        loss = F.mse_loss(val_sincos, tgt_sincos, reduction="none").mean(dim=(0, 2, 3))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError()


class PathToSinCosDecoder(nn.Module):
    """_summary_

    Args:
        z_dim (int, optional): _description_. Defaults to "8".
        aux_hidden_dim (int, optional): _description_. Defaults to "32".
    """

    def __init__(self, z_dim: int = 8, aux_hidden_dim: int = 32) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.aux_hidden_dim = aux_hidden_dim
        self.map = nn.Sequential(
            nn.Linear(z_dim, aux_hidden_dim), nn.Tanh(), nn.Linear(aux_hidden_dim, 2)
        )
    def extra_repr(self) -> str:
        return f"z_dim={self.z_dim}, aux_hidden_dim={self.aux_hidden_dim}"

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): _description_

        Returns:
            (Tensor): _description_
        """
        sincos = self.map(x)
        return sincos / sincos.norm(dim=-1, keepdim=True)


class ELBO(_Loss):
    """Class implementing the ELBO components.

    Args:
        reduction (str, optional): _description_. Defaults to "mean".
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.reduction = reduction

    def __repr__(self) -> str:
        return f"ELBO(reduction={self.reduction})"

    def forward(
        self,
        posterior: PathDistribution,
        prior: PathDistribution,
        likelihood: Distribution,
        evd_obs: Tensor,
        evd_tps: Tensor,
        evd_msk: Tensor,
        weights: Optional[dict[str, float]] = None
    ) -> Tuple[Tensor, dict[str, Tensor]]:
        """_summary_

        Args:
            posterior (PathDistribution): Approximate posterior path distribution :math:`q(z|x)`.
            prior (PathDistribution): Prior path distribution :math:`p(z)`.
            likelihood (Distribution): Distribution to compute log :math:`p(x|x)`.
            evd_obs (Tensor): (
                Evidence, e.g., [batch_size, evd_num_time_points, dim] for vector-valued evidence
                [batch_size, evd_num_time_points, 1, width, height] for image-valued evidence
            )
            evd_tps (Tensor): (
                Tensor of shape [batch_size, evd_num_time_points] that holds indices of the
                time points available in evd_obs
            )
            evd_msk (Tensor): Mask with :math:`{0,1}` values of same shape as evd_obs
            weights (dict[str, float], optional): (
                _description_.
                Defaults to "{ "pxz_weight": 1.0, "kl0_weight": 1.0, "klp_weight": 1.0 }".
            )
        Returns:
            Tuple[Tensor, dict[str, Tensor]]: _description_
        """
        if weights is None:
            weights = {
                "pxz_weight": 1.0,
                "kl0_weight": 1.0,
                "klp_weight": 1.0
            }

        # path KL (klp), initial state KL (kl0)
        klp, kl0 = kl_divergence(posterior, prior)

        mc_samples, _, num_tps, *_ = likelihood.mean.shape
        evd_obs, evd_msk = scatter_obs_and_msk(
            evd_obs, evd_msk, evd_tps, num_tps, mc_samples
        )  # [mc_samples, batch_len, num_tp, *per_tp_shape]

        log_pxz = -likelihood.log_prob(evd_obs)
        log_pxz[
            evd_msk == 0
        ] = 0  # -> [mc_samples, batch_len, num_time_points, *obs_shape]
        log_pxz = log_pxz.mean(dim=0)  # -> [batch_len, num_time_points, *obs_shape]

        log_pxz = reduce(log_pxz, "b ... -> b", "sum")  # -> [batch_len]
        numel = reduce(evd_msk[0], "b ... -> b", "sum")

        elbo = (
            weights["kl0_weight"] * kl0
            + weights["klp_weight"] * klp
            + weights["pxz_weight"] * log_pxz
        ) / numel  # -> [batch_len]

        if self.reduction == "mean":
            return elbo.mean(), {
                "kl0": kl0.mean(),
                "klp": klp.mean(),
                "log_pxz": log_pxz.mean(),
            }
        elif self.reduction == "sum":
            return elbo.sum(), {
                "kl0": kl0.sum(),
                "klp": klp.sum(),
                "log_pxz": log_pxz.sum(),
            }
        else:
            return elbo, {"kl0": kl0, "klp": klp, "log_pxz": log_pxz}


def default_SOnPathDistributionEncoder(
    h_dim: int,
    z_dim: int,
    n_deg: int,
    learnable_prior: bool = False,
    time_min: float = 0.0,
    time_max: float = 1.0,
) -> SOnPathDistributionEncoder:
    """_summary_

    Args:
        h_dim (int): _description_
        z_dim (int): _description_
        n_deg (int): _description_
        learnable_prior (bool, optional): _description_. Defaults to "False".
        time_min (float, optional): _description_. Defaults to "0.0".
        time_max (float, optional): _description_. Defaults to "1.0".

    Returns:
        SOnPathDistributionEncoder: _description_
    """
    group_dim = int(z_dim * (z_dim - 1) / 2)

    loc_map = nn.Linear(h_dim, z_dim)
    scl_map = nn.Linear(h_dim, 1)
    time_fn = Chebyshev(h_dim, n_deg, group_dim, time_min=time_min, time_max=time_max)
    return SOnPathDistributionEncoder(
        loc_map, scl_map, time_fn, learnable_prior=learnable_prior, in_dim=h_dim
    )


class ActivityRecogNetwork(nn.Module):
    """_summary_

        Args:
            mtan_input_dim (int, optional): _description_. Defaults to "32".
            mtan_hidden_dim (int, optional): _description_. Defaults to "32".
            use_atanh (bool, optional): _description_. Defaults to "False".
    """

    def __init__(
        self,
        mtan_input_dim: int = 32,
        mtan_hidden_dim: int = 32,
        use_atanh: bool = False
    ) -> None:
        super().__init__()
        self.mtan_input_dim = mtan_input_dim
        self.mtan_hidden_dim = mtan_hidden_dim
        self.use_atanh = use_atanh
        self.learn_emb = True
        self.mtan = MTANEncoder(
            input_dim=mtan_input_dim,
            query=torch.linspace(0, 1.0, 128),
            nhidden=mtan_hidden_dim,
            embed_time=128,
            num_heads=1,
            learn_emb=self.learn_emb
        )

    def extra_repr(self) -> str:
        return (
            f"mtan_input_dim={self.mtan_hidden_dim}, "
            f"mtan_hidden_dim={self.mtan_hidden_dim}, use_atanh={self.use_atanh}"
        )

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): _description_

        Returns:
            (Tensor): _description_
        """
        observed_data, observed_mask, observed_tp = x
        h = self.mtan(torch.cat((observed_data, observed_mask), 2), observed_tp)
        if self.use_atanh:
            eps = 1e-5
            h = h - h.sign() * eps
            h = h.atanh()
        return h
    
PhysioNetRecogNetwork = ActivityRecogNetwork


class GenericMLP(nn.Module):
    """_summary_

        Args:
            inp_dim (int): _description_
            out_dim (int): _description_
            n_hidden (int, optional): _description_. Defaults to "32".
            n_layers (int, optional): _description_. Defaults to "1".
    """

    def __init__(
            self,
            inp_dim: int,
            out_dim: int,
            n_hidden: int = 32,
            n_layers: int = 1
    )-> None:
        """_summary_

        Args:
            inp_dim (int): _description_
            out_dim (int): _description_
            n_hidden (int, optional): _description_. Defaults to "32".
            n_layers (int, optional): _description_. Defaults to "1".
        """
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        assert n_layers > 0, "Number of layers needs to be > 0"
        if n_layers == 1:
            self.map = nn.Linear(inp_dim, out_dim)
        else:
            layers = [nn.Linear(inp_dim, n_hidden)]
            layers.append(nn.ReLU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(n_hidden, n_hidden))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_hidden, out_dim))
            self.map = nn.Sequential(*layers)

    def extra_repr(self) -> str:
        return (
            f"inp_dim={self.inp_dim}, out_dim={self.out_dim}, "
            f"n_hidden={self.n_hidden}, n_layers={self.n_layers})"
        )

    def forward(self, inp: Tensor) -> Tensor:
        """_summary_

        Args:
            inp (Tensor): _description_

        Returns:
            (Tensor): _description_
        """
        return self.map(inp)


class ToyRecogNet(nn.Module):
    def __init__(self, h_dim:int=3):
        super().__init__()
        self.h = nn.Parameter(torch.rand(1,h_dim))
        
    def forward(self, x):
        obs, _, _ = x
        return self.h.repeat(obs.shape[0],1)


class ToyReconNet(nn.Module):
    def __init__(self, z_dim:int=3):
        super().__init__()
        self.map = nn.Linear(z_dim, 1)
    
    def forward(self, x):
        out = self.map(x)    
        return out