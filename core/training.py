"""Generic training routine used in ALL experiments.

Sebastian Zeng, Florian Graf, Roland Kwitt (2023)
"""

import numpy as np
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import scatter_obs_and_msk


def generic_train(args,
    dl: torch.utils.data.DataLoader, 
    modules: nn.ModuleDict, 
    elbo_loss: nn.Module,
    aux_loss: nn.Module, 
    optimizer: torch.optim.Optimizer,
    desired_t: torch.Tensor,
    device: str,
    aux_weight_mul: Optional[float] = None
    ):
    
    stats = defaultdict(list)

    modules.train()  
    for _, batch in enumerate(dl):
        parts = {key: val.to(device) for key, val in batch.items()}
        
        inp = (parts['inp_obs'], parts['inp_msk'], parts['inp_tps'])
        batch_len = parts['evd_obs'].shape[0]
                
        h = modules['recog_net'](inp)
        qzx, pz = modules['qzx_net'](h, desired_t)
        zis = qzx.rsample((args.mc_train_samples,))
        pxz = modules['pxz_net'](zis)
        if dl.dataset.has_aux: 
            aux = modules['aux_net'](zis)

        elbo_val, elbo_parts = elbo_loss(
            qzx,
            pz,
            pxz,
            parts['evd_obs'],
            parts['evd_tid'],
            parts['evd_msk'],
            {
                "kl0_weight": args.kl0_weight,
                "klp_weight": args.klp_weight,
                "pxz_weight": args.pxz_weight,
            },
        )
        if dl.dataset.has_aux:
            aux_val = aux_loss(aux, parts['aux_obs'], parts['aux_tid'])
            loss = elbo_val + args.aux_weight*aux_weight_mul*aux_val
        else:
            loss = elbo_val
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stats['loss'].append(loss.item()*batch_len)
        stats['elbo'].append(elbo_val.item()*batch_len)
        stats['kl0'].append(elbo_parts['kl0'].item()*batch_len)
        stats['klp'].append(elbo_parts['klp'].item()*batch_len)
        stats['log_pxz'].append(elbo_parts['log_pxz'].item()*batch_len)  
        if dl.dataset.has_aux:
            stats['aux_val'].append(aux_val.item()*batch_len)

    stats = {key: np.sum(val)/len(dl.dataset) for key, val in stats.items()}
    return stats


def evaluate_interpolation(args,
    dl: torch.utils.data.DataLoader, 
    modules: nn.ModuleDict, 
    elbo_loss: nn.Module,
    desired_t: torch.Tensor,
    ):
    """Evaluation code specific to Pendulum regression experiment."""
    stats = defaultdict(list)
    
    modules.eval()
    with torch.no_grad():
        for _, batch in enumerate(dl):
            parts = batch
            parts = {key: parts[key].to(args.device) for key in parts.keys()}
            inp = (parts['inp_obs'], parts['inp_msk'], parts['inp_tps'])
            
            h = modules['recog_net'](inp)
            qzx, pz = modules['qzx_net'](h, desired_t)
            zis = qzx.rsample((args.mc_eval_samples,))
            pxz = modules['pxz_net'](zis) 
        
            elbo_val, elbo_parts = elbo_loss(
                qzx,
                pz,
                pxz,
                parts['evd_obs'],
                parts['evd_tid'],
                parts['evd_msk'],
                {
                    "kl0_weight": args.kl0_weight,
                    "klp_weight": args.klp_weight,
                    "pxz_weight": args.pxz_weight,
                },
            )
            loss = elbo_val            

            mc_samples, batch_len, num_tps, *_ =  pxz.mean.shape

            _, inp_msk = scatter_obs_and_msk(
                parts['inp_obs'], parts['inp_msk'], parts['inp_tid'], num_tps, mc_samples
                )
            evd_obs, evd_msk = scatter_obs_and_msk(
                parts['evd_obs'], parts['evd_msk'], parts['evd_tid'], num_tps, mc_samples
                )
    
            impt_msk = ~inp_msk.bool() * evd_msk
            mse_full = F.mse_loss(evd_obs, pxz.mean, reduction='none')
            mse_impt = F.mse_loss(evd_obs, pxz.mean, reduction='none')
            
            mse_full = mse_full[evd_msk==1].sum() / evd_msk.sum()
            mse_impt = mse_impt[impt_msk==1].sum() / impt_msk.sum()

            stats['loss'].append(loss.item()*batch_len)
            stats['elbo'].append(elbo_val.item()*batch_len)
            stats['kl0'].append(elbo_parts['kl0'].item()*batch_len)
            stats['klp'].append(elbo_parts['klp'].item()*batch_len)
            stats['log_pxz'].append(elbo_parts['log_pxz'].item()*batch_len)    
            stats['mse_full'].append(mse_full.item()*batch_len) 
            stats['mse_impt'].append(mse_impt.item()*batch_len) 
        stats = {key: np.sum(val)/len(dl.dataset) for key, val in stats.items()}
        return stats