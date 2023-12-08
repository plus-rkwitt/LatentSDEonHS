"""Implementation of the geometric Euler-Maryuama SDE solver."""

import torch
from utils.misc import vec_to_matrix

def geometric_euler(z0, drift, cov, dt, basis):    
    noise = torch.randn(z0.shape[:-2]+drift.shape, device=z0.device)
    if cov.numel() == 1:
        noise = noise * cov
    else:
        noise = torch.einsum('btij, ...btj -> ...bti', cov, noise)
    
    noise = torch.einsum('...td, t -> ...td', noise, dt.sqrt())
    drift = torch.einsum('...td, t -> ...td', drift, dt)
    omegas = drift + noise
    omegas = vec_to_matrix(omegas, basis)

    Qs = torch.matrix_exp(omegas.contiguous())
    zi = [z0]
    for t_idx in range(len(dt)):
        Qt = Qs[...,t_idx,:,:]
        Qtz = torch.einsum('...ij, ...j -> ...i', Qt, zi[-1])
        zi.append(Qtz)
    zi = torch.stack(zi, dim=-2)
    return zi