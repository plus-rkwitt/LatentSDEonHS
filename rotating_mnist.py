"""RotatingMNIST (i.e., rotating 3's) experiment from Sec. 4.2 of

    Zeng S., Graf F. and Kwitt, R.
    Latent SDEs on Homogeneous Spaces
    NeurIPS 2023

Sebastian Zeng, Florian Graf and Roland Kwitt (2023)
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


from data.mnist_provider import RotatingMNISTProvider
from core.training import generic_train
from core.models import (
    ELBO,
    RotatingMNISTRecogNetwork,
    RotatingMNISTReconNetwork,
    PathToBernoulliDecoder,
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
from utils.parser import generic_parser, remove_argument



def extend_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("Experiment specific arguments")
    group.add_argument("--n-filters", type=int, default=8)
    return parser


def evaluate(
    args,
    dl: torch.utils.data.DataLoader,
    modules: nn.ModuleDict,
    elbo_loss: nn.Module,
    desired_t: torch.Tensor,
    device: str,
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

            rec = pxz.probs.mean(0, keepdim=True)
            _, batch_len, num_time_points, *rec_shape = rec.shape
            sz0 = torch.Size([1, batch_len, num_time_points]) + torch.Size(
                [1] * len(rec_shape)
            )
            sz1 = torch.Size([1, 1, 1]) + torch.Size(rec_shape)
            tpidx = parts["evd_tid"].view(sz0).repeat(sz1).long()
            rec = rec.gather(2, tpidx)

            target_idx = 3
            mse = (rec - parts["evd_obs"].view_as(rec)).square().mean()
            mse_on_target = (
                (
                    rec[:, :, target_idx]
                    - parts["evd_obs"].view_as(rec)[:, :, target_idx]
                )
                .square()
                .mean()
            )

            loss = elbo_val
            stats["loss"].append(loss.item() * batch_len)
            stats["elbo"].append(elbo_val.item() * batch_len)
            stats["kl0"].append(elbo_parts["kl0"].item() * batch_len)
            stats["klp"].append(elbo_parts["klp"].item() * batch_len)
            stats["log_pxz"].append(elbo_parts["log_pxz"].item() * batch_len)
            stats["mse_full"].append(mse.item() * batch_len)
            stats["mse_trgt"].append(mse_on_target.item() * batch_len)
    stats = {key: np.sum(val) / len(dl.dataset) for key, val in stats.items()}
    return stats


def main():
    experiment_id = int(SystemRandom().random() * 100000)
    parser = extend_argparse(generic_parser)
    remove_argument(parser, "--freeze-sigma") # no PathToGaussianDecoder used here
    args = parser.parse_args()
    print(args)

    check_logging_and_checkpointing(args)

    set_up_logging(
        console_log_level=args.loglevel,
        console_log_color=True,
        logfile_file=os.path.join(args.log_dir, f"rotating_mnist_{experiment_id}.txt")
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

    provider = RotatingMNISTProvider(args.data_dir, download=True, random_state=133)
    dl_trn = provider.get_train_loader(batch_size=args.batch_size, shuffle=True)
    dl_val = provider.get_val_loader(batch_size=args.batch_size, shuffle=False)
    dl_tst = provider.get_test_loader(batch_size=args.batch_size, shuffle=False)

    desired_t = torch.linspace(0, 0.99, provider.num_timepoints, device=args.device)

    logging.debug(f"Number of training samples={provider.num_train_samples}")
    logging.debug(f"Number of validation samples={provider.num_val_samples}")
    logging.debug(f"Number of testing samples={provider.num_test_samples}")

    recog_net = RotatingMNISTRecogNetwork(n_filters=args.n_filters)
    recon_net = RotatingMNISTReconNetwork(
        z_dim=args.z_dim, n_filters=args.n_filters * 2
    )
    qzx_net = default_SOnPathDistributionEncoder(
        h_dim=256, z_dim=args.z_dim, n_deg=args.n_deg, time_min=0.0, time_max=20.0
    )
    pxz_net = PathToBernoulliDecoder(logit_map=recon_net)

    modules = nn.ModuleDict(
        {
            "recog_net": recog_net,
            "recon_net": recon_net,
            "pxz_net": pxz_net,
            "qzx_net": qzx_net,
        }
    )
    modules = modules.to(args.device)

    optimizer = optim.AdamW(modules.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.restart, eta_min=0, last_epoch=-1, verbose=False
    )

    logging.debug(f"Number of model parameters={count_parameters(modules)}")

    elbo_loss = ELBO(reduction="mean")

    stats = defaultdict(list)
    stats_mask = {
        "trn": ["log_pxz", "loss"],
        "tst": ["loss", "mse_full", "mse_trgt", "mse_trgt*"],
        "val": ["loss", "mse_full", "mse_trgt", "mse_trgt*"],
        "oth": ["lr"],
    }
    pm = ProgressMessage(stats_mask)

    best_epoch_val_mse_on_target = np.inf
    best_epoch_tst_mse_on_target = np.inf

    for epoch in range(1, args.n_epochs + 1):
        trn_stats = generic_train(
            args, dl_trn, modules, elbo_loss, None, optimizer, desired_t, args.device
        )
        tst_stats = evaluate(args, dl_tst, modules, elbo_loss, desired_t, args.device)
        val_stats = evaluate(args, dl_val, modules, elbo_loss, desired_t, args.device)

        stats["oth"].append({"lr": scheduler.get_last_lr()[-1]})
        scheduler.step()

        if val_stats["mse_trgt"] < best_epoch_val_mse_on_target:
            best_epoch_val_mse_on_target = val_stats["mse_trgt"]
            best_epoch_tst_mse_on_target = tst_stats["mse_trgt"]
            save_checkpoint(
                args,
                'best',
                experiment_id,
                modules,
                desired_t,
            )
        val_stats["mse_trgt*"] = best_epoch_val_mse_on_target
        tst_stats["mse_trgt*"] = best_epoch_tst_mse_on_target

        stats["trn"].append(trn_stats)
        stats["tst"].append(tst_stats)
        stats["val"].append(val_stats)

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
            fname = os.path.join(args.log_dir, f"rotating_mnist_{experiment_id}.json")
            save_stats(args, stats, fname)


if __name__ == "__main__":
    main()
