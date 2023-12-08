"""PhysioNet 2012 interpolation experiment from Sec. 4.2 of

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

from data.physionet_provider import PhysioNetProvider
from core.models import (
    ELBO,
    PhysioNetRecogNetwork,
    GenericMLP,
    PathToGaussianDecoder,
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
    group.add_argument("--n-dec-layers", type=int, default=1)
    group.add_argument("--dec-hidden-dim", type=int, default=64)
    group.add_argument("--use-atanh", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--sample-tp", type=float, default=0.5)
    group.add_argument("--quantization", type=float, default=0.1)
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
        logfile_file=os.path.join(args.log_dir, f"physionet_interpolation_{args.sample_tp}_{experiment_id}.txt")
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

    provider = PhysioNetProvider(args.data_dir, quantization=args.quantization, sample_tp=args.sample_tp)
    dl_trn = provider.get_train_loader(
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True)
    dl_tst = provider.get_test_loader(
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True)

    desired_t = torch.linspace(0, 1.0, provider.num_timepoints, device=args.device)

    logging.debug(f"Number of training samples={provider.num_train_samples}")
    logging.debug(f"Number of testing samples={provider.num_test_samples}")

    recog_net = PhysioNetRecogNetwork(
        mtan_input_dim=provider.input_dim,
        mtan_hidden_dim=args.h_dim,
        use_atanh=args.use_atanh
    )
    recon_net = GenericMLP(
        inp_dim=args.z_dim,
        out_dim=provider.input_dim,
        n_hidden=args.dec_hidden_dim,
        n_layers=args.n_dec_layers,
    )
    # parametrizes q(z|x)
    qzx_net = default_SOnPathDistributionEncoder(
        h_dim=args.h_dim,
        z_dim=args.z_dim,
        n_deg=args.n_deg,
        learnable_prior=args.learnable_prior,
        time_min=0.0,
        time_max=2.0 * desired_t[-1].item(),
    )
    # parametrizes p(x|z)
    pxz_net = PathToGaussianDecoder(mu_map=recon_net, sigma_map=None, initial_sigma=0.1)
    if args.freeze_sigma:
        logging.debug("Use frozen sigma when computing PathToGaussianDecoder")
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

    # logging
    stats = defaultdict(list)
    stats_mask = {
        "trn": ["log_pxz", "loss"],
        "tst": ["loss", "mse_full", "mse_impt"],
        "oth": ["lr"],
    }
    pm = ProgressMessage(stats_mask)

    for epoch in range(1, args.n_epochs + 1):
        trn_stats = generic_train(
            args, dl_trn, modules, elbo_loss, None, optimizer, desired_t, args.device
        )
        tst_stats = evaluate(args, dl_tst, modules, elbo_loss, desired_t)

        stats["trn"].append(trn_stats)
        stats["tst"].append(tst_stats)
        stats["oth"].append({"lr": scheduler.get_last_lr()[-1]})

        scheduler.step()

        if args.checkpoint_at and (epoch in args.checkpoint_at):
            save_checkpoint(
                args,
                epoch,
                experiment_id,
                modules,
                desired_t,
            )

        msg = pm.build_progress_message(stats, epoch)
        logging.debug(msg)

        if args.enable_file_logging:
            fname = os.path.join(
                args.log_dir, f"physionet_interpolation_{args.sample_tp}_{experiment_id}.json"
            )
            save_stats(args, stats, fname)


if __name__ == "__main__":
    main()
