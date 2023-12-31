"""Implements the PathDistribution objects and KL divergences among them.
"""

import torch
from torch import Tensor
from typing import Callable, Tuple

from torch.distributions import (
    Distribution, 
    register_kl, 
    kl_divergence, 
    Normal)
from core.power_spherical.distributions import PowerSpherical, HypersphericalUniform
from core.sde_solvers import geometric_euler
from utils.misc import vec_to_matrix


class PathDistribution(Distribution):
    """Base class for distributions of paths generated by an SDE 
    on a homogeneous space of the form

        z_t = K_t z_t dt + \sum_i d\omega_i  V_{i,t} z_t  + \sum_i d\omega_i  V_{i,t}^2 z_t dt

    where K_t and V_{i,t} are tangent vector fields in the Lie algebra.

    Args:
        p0 (Distribution):
            Distribution for starting points
        K (function):
            Mapping for t |-> K_t, parametrizing the drift
        sigma (Tensor):
            scales noise in path
        t (Tensor):
            Tensor of time points
        kl_samples (int):
            Number of samples used to compute KL divergence.
            Defaults to 1.
        validate_args (Bool):
            Validate argument constraints. Defaults to False.
        solver (Callable):
            SDE solver to use. Defaults to core.sde_solvers.geometric_euler.
    """

    arg_constraints = {
        "sigma": torch.distributions.constraints.positive,
        "t": torch.distributions.constraints.nonnegative,
    }
    has_rsample = True

    def __init__(
        self,
        p0: Distribution,
        K: Callable[[Tensor], Tensor],
        sigma: Tensor,
        t: Tensor,
        validate_args=False,
        kl_samples: int = 1,
        solver: Callable = geometric_euler
        ) -> None:

        assert p0.sample().device == sigma.device == t.device, 'Device mismatch'
        self.device = sigma.device

        batch_shape = p0.sample().shape[0:1]
        event_shape = t.shape + p0.sample().shape[1:2]
        super(PathDistribution, self).__init__(batch_shape = batch_shape, event_shape=event_shape, validate_args=validate_args)

        self.p0 = p0
        self._K = K
        self.sigma = sigma.flatten()
        self.t = t
        self.kl_samples = kl_samples
        self.group_dim, self.basis = self._generate_basis()
        self._solver = solver

    @property
    def t(self) -> Tensor:
        return self._t
    @t.setter
    def t(self, val: Tensor) -> None:
        self._t = val
        self.dt = torch.diff(self.t)
        self._Kt = self._K(self.t[:-1])

    @property
    def K(self) -> Callable[[Tensor], Tensor]:
        return self._K
    @K.setter
    def K(self, val: Callable[[Tensor], Tensor]):
        self._K = val
        self._Kt = self._K(self.t[:-1])

    @property
    def steps(self) -> int:
        return len(self.t)

    @property
    def Kt(self) -> Tensor:
        return self._Kt

    @property
    def dim(self) -> int:
        if hasattr(self.p0, 'dim'):
            return self.p0.dim
        else:
            return self.event_shape[-1]

    def _generate_basis(self) -> Tuple[int, Tensor]:
        """Returns the dim and a basis of the Lie Algebra of the Lie group.
        """
        raise NotImplementedError(f'_generate basis not implemented for class {type(self).__name__}')

    def _integrate(self, z0: Tensor) -> Tensor:
        """Integrates the SDE forward, starting at the initial value z0.
        """
        return self._solver(z0, self.Kt, self.sigma, self.dt, self.basis)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draw sample from path distribution.
        """
        z0 = self.p0.rsample(sample_shape)
        zi = self._integrate(z0)
        self.zi = zi
        return zi

    def __repr__(self) -> str:
        return f"{type(self).__name__}(batch_shape={list(self.batch_shape)}, steps={self.steps}, dim={self.dim})"


@register_kl(PathDistribution, PathDistribution)
def _kl_pathdistributions(p: PathDistribution, q:PathDistribution) -> Tuple[Tensor, Tensor]:
    assert torch.all(p.basis == q.basis) and p.sigma == q.sigma, "Diffusion terms do not agree"
    kl0 = kl_divergence(p.p0, q.p0) # kl-div of initial values

    zi = p.zi if hasattr(p, 'zi') else p.rsample((p.kl_samples,))
    diffusion = torch.einsum('dij, ...btj -> ...btdi', p.basis, zi[...,:-1,:]) * p.sigma
    diffusion = torch.einsum('...btdi, ...btdj -> ...btij', diffusion, diffusion)
    diff_pinv = torch.linalg.pinv(diffusion)

    delta_drift = vec_to_matrix(p.Kt-q.Kt, p.basis)
    proj_drift = torch.einsum('...btij, ...btj -> ...bti ',delta_drift,zi[...,:-1,:])
    kl_path = torch.einsum('...i, ...ij, ...j -> ...', proj_drift, diff_pinv, proj_drift)

    kl_path = torch.einsum('...nt,t -> ...n', kl_path, p.dt) # average over time
    kl_path = kl_path.mean(dim=0) # average over samples
    return kl_path, kl0


class SOnPathDistribution(PathDistribution):
    """Distribution of paths generated by an SDE of the form

        z_t = K_t z_t dt + \sum_i d\omega_i  V_{i,t} z_t  + \sum_i d\omega_i  V_{i,t}^2 z_t dt

    Args:
        p0 (PowerSpherical):
            Distribution for starting points
        K (Callable):
            Mapping for t |-> K_t, corresponds to the drift
        sigma (Tensor):
            scales noise in path
        t (Tensor):
            Tensor of time points
        kl_samples (int)
            Number of samples used to compute KL divergence. Defaults to 1.
        validate_args (Bool):
            Validate argument constraints. Defaults to False.
        solver (Callable):
            SDE solver to use. Defaults to core.sde_solvers.geometric_euler.

    Example:
        location = torch.rand(5, 3)
        location = torch.nn.functional.normalize(location)
        scale = torch.tensor([50., 50., 50., 50, 50.], dtype=torch.float32)
        pz0 = PowerSpherical(location, scale)
        dummy = torch.randn(5, 10)
        timefn = Chebyshev(10, 4, 3)
        def K(t): return timefn(dummy, t)
        psd = PathOnSphereDistribution(pz0, K, torch.tensor([0.5]), torch.linspace(0,1,10))
    """

    arg_constraints = {
        "sigma": torch.distributions.constraints.positive,
        "t": torch.distributions.constraints.nonnegative,
    }
    has_rsample = True

    def __init__(
        self,
        p0: PowerSpherical,
        K: Callable[[Tensor], Tensor],
        sigma: Tensor,
        t: Tensor,
        validate_args=False,
        kl_samples: int = 1,
        solver: Callable = geometric_euler
        ) -> None:
        super().__init__(p0, K ,sigma, t, validate_args, kl_samples, solver)

    def _generate_basis(self) -> Tuple[int, Tensor]:
        """Generates the basis of so(n) that consists of the matrices e_i * e_j^T e_j * e_i^T.
        """
        group_dim = int(self.dim * (self.dim-1)/2)
        idx = torch.tril_indices(row=self.dim, col=self.dim, offset=-1)
        basis =  torch.zeros(group_dim,self.dim,self.dim, device=self.device)
        basis [:,idx[0],idx[1]]=torch.eye(group_dim, device=self.device)
        basis = basis - basis.permute(0,2,1)
        return group_dim, basis


class BrownianMotionOnSphere(SOnPathDistribution):
    """Implements an uninformative prior on the sphere."""
    def __init__(
        self,
        dim,
        sigma: Tensor,
        t: Tensor,
        validate_args=False,
        kl_samples: int = 1,
        solver: Callable = geometric_euler
        ) -> None:

        p0 = HypersphericalUniform(dim, device=sigma.device, validate_args=validate_args)
        group_dim = int(dim*(dim-1)/2)
        def K(tt: Tensor): 
            return torch.zeros(len(tt), group_dim, device=sigma.device)
        super().__init__(p0, K, sigma, t, validate_args, kl_samples, solver)


@register_kl(HypersphericalUniform, HypersphericalUniform)
def _kl_hyperspherical_uniform(p, q):
    return torch.zeros(1)


@register_kl(SOnPathDistribution, SOnPathDistribution)
def _kl_pathdistributions_son(p:SOnPathDistribution, q: SOnPathDistribution) -> Tuple[Tensor, Tensor]:
    assert torch.all(p.basis == q.basis) and p.sigma == q.sigma, "Diffusion terms do not agree"
    kl0 = kl_divergence(p.p0, q.p0) # kl-div of initial values

    drift = vec_to_matrix(p.Kt-q.Kt, p.basis)
    zi = p.zi if hasattr(p, 'zi') else p.rsample((p.kl_samples,))
    projection = torch.einsum('ntij, ...ntj -> ...nti ',drift,zi[...,:-1,:])
    norm_squared = projection.square().sum(dim=(-1)) # sum over spatial dimensions

    kl_path = torch.einsum('...nt,t -> ...n', norm_squared, p.dt) / p.sigma.square()  # average over time
    kl_path = kl_path.mean(dim=0) # average over samples
    return kl_path, kl0


class GLnPathDistribution(PathDistribution):
    """Distribution of paths generated by an SDE of the form

        z_t = K_t z_t dt + \sum_i d\omega_i  V_{i,t} z_t

        Args:
            p0 (torch.distributions.Normal):
                Distribution for starting points
            K (function):
                Mapping for t |-> K_t, corresponds to the drift
            sigma (Tensor):
                scales noise in path
            t (Tensor):
                Tensor of time points
            kl_samples (int)
                Number of samples used to compute KL divergence. 
                Defaults to 1.
            validate_args (Bool):
                Validate argument constraints. Defaults to False.
            solver (Callable):
                SDE solver to use. Defaults to core.sde_solvers.geometric_euler.
    """

    arg_constraints = {
        "sigma": torch.distributions.constraints.positive,
        "t": torch.distributions.constraints.nonnegative,
    }
    has_rsample = True

    def __init__(
        self,
        p0: Normal,
        K: Callable[[Tensor], Tensor],
        sigma: Tensor,
        t: Tensor,
        validate_args=False,
        kl_samples: int = 1,
        solver: Callable = geometric_euler
        ) -> None:
        super(GLnPathDistribution, self).__init__(p0, K ,sigma, t, validate_args, kl_samples, solver)

    def _generate_basis(self) -> Tuple[int, Tensor]:
        group_dim = self.dim * self.dim
        basis = torch.eye(group_dim, group_dim).view(group_dim, self.dim, self.dim)
        return group_dim, basis


@register_kl(GLnPathDistribution, GLnPathDistribution)
def _kl_pathdistributions_gln(p: GLnPathDistribution, q: GLnPathDistribution) -> Tuple[Tensor, Tensor]:
    assert torch.all(p.basis == q.basis) and p.sigma == q.sigma, "Diffusion terms do not agree"
    kl0 = kl_divergence(p.p0, q.p0).sum(dim=-1) # kl-div of initial values

    drift = vec_to_matrix(p.Kt-q.Kt, p.basis)
    zi = p.zi if hasattr(p, 'zi') else p.rsample((p.kl_samples,))
    projection = torch.einsum('ntij, ...ntj -> ...nti ',drift,zi[...,:-1,:])
    norm_squared = projection.square().sum(dim=(-1)) # sum over spatial dimensions
    kl_path = torch.einsum('...nt,t -> ...n', norm_squared, p.dt) / p.sigma.square()  # average over time
    kl_path = kl_path.mean(dim=0) # average over samples
    return kl_path, kl0