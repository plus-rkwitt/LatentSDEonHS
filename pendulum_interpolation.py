"""Pendulum interpolation experiment from Sec. 4.2 of

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
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.pendulum_provider import PendulumProvider
from core.models import (
    PendulumRecogNetwork,
    PendulumReconNetwork,
    PathToGaussianDecoder,
    ELBO,
    default_SOnPathDistributionEncoder,
)
from core.training import generic_train
from core.training import evaluate_interpolation as evaluate
from utils.logger import set_up_logging
from utils.misc import (
    set_seed,
    count_parameters,
    save_checkpoint,
    ProgressMessage,
    save_stats,
    check_logging_and_checkpointing
)
from utils.parser import generic_parser


def extend_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("Experiment specific arguments")
    group.add_argument("--use-atanh", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main():
    experiment_id = int(SystemRandom().random() * 100000)
    parser = extend_argparse(generic_parser)
    args = parser.parse_args()
    print(args)
    
    check_logging_and_checkpointing(args)

    set_up_logging(
        console_log_level=args.loglevel,
        console_log_color=True,
        logfile_file=os.path.join(args.log_dir, f"pendulum_interpolation_{experiment_id}.txt")
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

    provider = PendulumProvider(args.data_dir, "interpolation")
    dl_trn = provider.get_train_loader(
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=None,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    dl_tst = provider.get_test_loader(
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=None,
        num_workers=8,
        pin_memory=True,
    )
    dl_val = provider.get_val_loader(
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=None,
        num_workers=8,
        pin_memory=True,
    )

    desired_t = torch.linspace(0, 0.99, provider.num_timepoints, device=args.device)

    recog_net = PendulumRecogNetwork(
        h_dim=args.h_dim, mtan_input_dim=args.h_dim
    )
    recon_net = PendulumReconNetwork(z_dim=args.z_dim)
    pxz_net = PathToGaussianDecoder(mu_map=recon_net, sigma_map=None) 
    qzx_net = default_SOnPathDistributionEncoder(
        h_dim=args.h_dim,
        z_dim=args.z_dim,
        n_deg=args.n_deg,
        learnable_prior=args.learnable_prior,
        time_min=0.0,
        time_max=2.0 * desired_t[-1].item(),
    )
    if args.freeze_sigma:
        logging.debug("Froze sigma when computing PathToGaussianDecoder")
        pxz_net.sigma.requires_grad = False

    modules = nn.ModuleDict(
        {
            "recog_net": recog_net,
            "recon_net": recon_net,
            "pxz_net": pxz_net,
            "qzx_net": qzx_net,
        }
    )
    modules = modules.to(args.device)

    optimizer = optim.Adam(modules.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, args.restart, eta_min=0, last_epoch=-1, verbose=False)

    logging.debug(f"Number of model parameters={count_parameters(modules)}")

    elbo_loss = ELBO(reduction="mean")

    stats = defaultdict(list)
    stats_mask = {
        "trn": ["log_pxz", "loss"],
        "tst": ["loss", "mse_full", "mse_impt", "mse_impt*"],
        "oth": ["lr"],
    }
    pm = ProgressMessage(stats_mask)

    best_epoch_val_mse_impt = np.inf
    best_epoch_tst_mse_impt = np.inf

    for epoch in range(1, args.n_epochs + 1):
        trn_stats = generic_train(args, dl_trn, modules, elbo_loss, None, optimizer, desired_t, args.device)
        tst_stats = evaluate(args, dl_tst, modules, elbo_loss, desired_t)
        val_stats = evaluate(args, dl_val, modules, elbo_loss, desired_t)

        stats["oth"].append({"lr": scheduler.get_last_lr()[-1]})
        scheduler.step()

        if val_stats["mse_impt"] < best_epoch_val_mse_impt:
            best_epoch_val_mse_impt = val_stats["mse_impt"]
            best_epoch_tst_mse_impt = tst_stats["mse_impt"]
            save_checkpoint(
                args, 
                'best', 
                experiment_id, 
                modules, 
                desired_t)
        val_stats["mse_impt*"] = best_epoch_val_mse_impt
        tst_stats["mse_impt*"] = best_epoch_tst_mse_impt

        stats["trn"].append(trn_stats)
        stats["tst"].append(tst_stats)
        stats["val"].append(val_stats)

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
            fname = os.path.join(args.log_dir, f"pendulum_interpolation_{experiment_id}.json")
            save_stats(args, stats, fname)


if __name__ == "__main__":
    main()
