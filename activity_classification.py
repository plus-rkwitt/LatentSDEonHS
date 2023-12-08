"""Human activity classification experiment from Sec. 4.1 of

    Zeng S., Graf F. and Kwitt, R.
    Latent SDEs on Homogeneous Spaces
    NeurIPS 2023
"""

import os
import logging
import argparse
import numpy as np
from random import SystemRandom
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.activity_provider import HumanActivityProvider
from core.training import generic_train
from core.models import (
    ELBO,
    ActivityRecogNetwork,
    GenericMLP,
    PathToGaussianDecoder,
    PerTimePointCrossEntropyLoss,
    default_SOnPathDistributionEncoder,
)

from utils.logger import set_up_logging
from utils.misc import (
    set_seed,
    count_parameters,
    ProgressMessage,
    save_checkpoint,
    save_stats,
    check_logging_and_checkpointing
)
from utils.parser import generic_parser


def extend_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("Experiment specific arguments")
    group.add_argument("--aux-weight", type=float, default=10.)
    group.add_argument("--aux-hidden-dim", type=int, default=32)
    group.add_argument("--use-atanh", action=argparse.BooleanOptionalAction, default=False)
    return parser


def compute_accuracy(input, target, tps):
    mc_samples, input_batch_len, input_num_time_points, dim = input.shape
    target_batch_len, target_num_time_points = target.shape

    assert input_batch_len == target_batch_len
    batch_len = input_batch_len

    idx = (
        tps.view([1, batch_len, target_num_time_points, 1])
        .repeat(mc_samples, 1, 1, dim)
        .long()
    )
    input_at_tps = input.gather(2, idx).mean(dim=0)
    return (input_at_tps.flatten(0, 1).argmax(1) == target.flatten()).float().mean()


def evaluate(
    args,
    dl: torch.utils.data.DataLoader,
    modules: nn.ModuleDict,
    elbo_loss: nn.Module,
    aux_loss: nn.Module,
    desired_t: torch.Tensor,
    device: str,
    aux_weight_mul: float
):
    stats = defaultdict(list)

    modules.eval()
    with torch.no_grad():
        for _, batch in enumerate(dl):
            parts = {key: val.to(device) for key, val in batch.items()}

            inp = (parts["inp_obs"], parts["inp_msk"], parts["inp_tps"])

            h = modules["recog_net"](inp)
            qzx, pz = modules["qzx_net"](h, desired_t)
            zis = qzx.rsample((args.mc_eval_samples,))
            pxz = modules["pxz_net"](zis)
            aux = modules["aux_net"](zis)

            elbo_val, elbo_parts = elbo_loss(
                qzx,
                pz,
                pxz,
                parts["evd_obs"],
                parts["evd_tid"],
                parts["evd_msk"],
                {
                    "kl0_weight": args.kl0_weight,
                    "klp_weight": args.klp_weight,
                    "pxz_weight": args.pxz_weight,
                },
            )
            aux_val = aux_loss(aux, parts["aux_obs"], parts["aux_tid"])
            loss = elbo_val + args.aux_weight * aux_weight_mul * aux_val
            aux_acc = compute_accuracy(aux, parts["aux_obs"], parts["aux_tid"])

            batch_len = parts["evd_obs"].shape[0]
            stats["loss"].append(loss.item() * batch_len)
            stats["aux_val"].append(aux_val.item() * batch_len)
            stats["aux_acc"].append(aux_acc.item() * batch_len)
            stats["elbo_val"].append(elbo_val.item() * batch_len)
            stats["elbo_kl0"].append(elbo_parts["kl0"].item() * batch_len)
            stats["elbo_klp"].append(elbo_parts["klp"].item() * batch_len)
            stats["elbo_log_pxz"].append(elbo_parts["log_pxz"].item() * batch_len)
        stats = {key: np.sum(val)/len(dl.dataset) for key, val in stats.items()}    
        return stats


def main():
    experiment_id = int(SystemRandom().random() * 100000)
    parser = extend_argparse(generic_parser)
    args = parser.parse_args()
    print(args)
    
    check_logging_and_checkpointing(args)
    
    set_up_logging(
        console_log_level=args.loglevel,
        console_log_color=True,
        logfile_file=os.path.join(args.log_dir, f"activity_classification_{experiment_id}.txt")
        if args.log_dir is not None
        else None,
        logfile_log_level=args.loglevel,
        logfile_log_color=False,
        log_line_template="%(color_on)s[%(created)d] [%(levelname)-8s] %(message)s%(color_off)s",

    )

    logging.debug(f"Experiment ID={experiment_id}")
    if args.seed > 0:
        set_seed(args.seed)
    logging.debug(f"Seed set to {args.seed}")

    provider = HumanActivityProvider(args.data_dir)
    dl_trn = provider.get_train_loader(batch_size=args.batch_size, shuffle=True)
    dl_val = provider.get_val_loader(batch_size=args.batch_size, shuffle=False)
    dl_tst = provider.get_test_loader(batch_size=args.batch_size, shuffle=False)

    desired_t = torch.linspace(0, 0.99, provider.num_timepoints, device=args.device)

    logging.debug(f"Number of training samples={provider.num_train_samples}")
    logging.debug(f"Number of validation samples={provider.num_val_samples}")
    logging.debug(f"Number of testing samples={provider.num_test_samples}")

    recog_net = ActivityRecogNetwork(
        mtan_input_dim=provider.input_dim, 
        mtan_hidden_dim=args.h_dim, 
        use_atanh=args.use_atanh)
    recon_net = GenericMLP(
        inp_dim=args.z_dim, 
        out_dim=provider.input_dim, 
        n_layers=1)
    qzx_net = default_SOnPathDistributionEncoder(
        h_dim=args.h_dim,
        z_dim=args.z_dim,
        n_deg=args.n_deg,
        learnable_prior=args.learnable_prior,
        time_min=0.0,
        time_max=2.0 * desired_t[-1].item(),
    )
    pxz_net = PathToGaussianDecoder(mu_map=recon_net, sigma_map=None, initial_sigma=1.0)
    aux_net = GenericMLP(
        inp_dim=args.z_dim, 
        out_dim=provider.num_classes, 
        n_hidden=args.aux_hidden_dim,
        n_layers=1)
    
    modules = nn.ModuleDict(
        {
            "recog_net": recog_net,
            "recon_net": recon_net,
            "pxz_net": pxz_net,
            "qzx_net": qzx_net,
            "aux_net": aux_net,
        }
    )
    modules = modules.to(args.device)

    optimizer = optim.Adam(modules.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, args.restart, eta_min=0, last_epoch=-1, verbose=False)

    logging.debug(f"Number of model parameters={count_parameters(modules)}")

    elbo_loss = ELBO(reduction="mean")
    aux_loss = PerTimePointCrossEntropyLoss(reduction="mean")

    stats = defaultdict(list)
    stats_mask = {
            'trn': ['log_pxz', 'loss'],
            'tst': ['loss', 'aux_val', 'aux_acc*'],
            'val': ['loss', 'aux_val', 'aux_acc*'],
            'oth': ['lr']
        }
    pm = ProgressMessage(stats_mask)

    args_dict = vars(args)

    best_epoch_val_acc = 0. # best validation accuracy
    best_epoch_tst_acc = 0. # best testing accuracy

    for epoch in range(1, args.n_epochs + 1):
        aux_weight_mul = (epoch / 60) ** 2 if epoch < 60 else 1

        trn_stats = generic_train(
            args,
            dl_trn,
            modules,
            elbo_loss,
            aux_loss,
            optimizer,
            desired_t,
            args.device,
            aux_weight_mul
        )
        tst_stats = evaluate(
            args, dl_tst, modules, elbo_loss, aux_loss, desired_t, args.device, aux_weight_mul
        )
        val_stats = evaluate(
            args, dl_val, modules, elbo_loss, aux_loss, desired_t, args.device, aux_weight_mul
        )

        stats['oth'].append({'lr': scheduler.get_last_lr()[-1]})
        scheduler.step()

        if val_stats['aux_acc'] > best_epoch_val_acc:
            best_epoch_val_acc = val_stats['aux_acc'] 
            best_epoch_tst_acc = tst_stats['aux_acc'] 
            save_checkpoint(
                args, 
                'best', 
                experiment_id, 
                modules, 
                desired_t)
        val_stats['aux_acc*'] = best_epoch_val_acc
        tst_stats['aux_acc*'] = best_epoch_tst_acc

        stats['trn'].append(trn_stats)
        stats['tst'].append(tst_stats)
        stats['val'].append(val_stats)

        if args.checkpoint_at and (epoch in args.checkpoint_at):
            save_checkpoint(
                args, 
                epoch, 
                experiment_id, 
                modules, 
                desired_t)

        msg = pm.build_progress_message(stats, epoch)
        logging.debug(msg)

        if args.enable_file_logging:
            fname = os.path.join(args.log_dir, f'activity_classification_{experiment_id}.json')
            save_stats(args, stats, fname)
    

if __name__ == "__main__":
    main()
