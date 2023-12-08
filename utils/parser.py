"""Setup some generic commandline arguments, common to all experiments."""

import argparse

"""Setup command-line parsing."""
generic_parser = argparse.ArgumentParser()
group0 = generic_parser.add_argument_group('Data loading/saving arguments')
group0.add_argument("--data-dir", type=str, default="data_dir")
group0.add_argument("--enable-file-logging", action=argparse.BooleanOptionalAction, default=False)
group0.add_argument("--log-dir", type=str, default=None)
group0.add_argument("--enable-checkpointing", action=argparse.BooleanOptionalAction, default=False)
group0.add_argument("--checkpoint-dir", type=str, default=None)
group0.add_argument("--checkpoint-at", type=int, nargs='*', default=[])

group1 = generic_parser.add_argument_group('Training arguments')
group1.add_argument("--batch-size", type=int, default=64)
group1.add_argument("--lr", type=float, default=1e-3)
group1.add_argument("--n-epochs", type=int, default=990)
group1.add_argument("--kl0-weight", type=float, default=1e-4)
group1.add_argument("--klp-weight", type=float, default=1e-4)
group1.add_argument("--pxz-weight", type=float, default=1.)
group1.add_argument("--seed", type=int, default=-1)
generic_parser.add_argument("--restart", type=int, default=30)
group1.add_argument("--device", type=str, default="cuda:0")

group2 = generic_parser.add_argument_group('Model configuration arguments')
group2.add_argument("--z-dim", type=int, default=16)
group2.add_argument("--h-dim", type=int, default=32)
group2.add_argument("--n-deg", type=int, default=4)
group2.add_argument("--learnable-prior", action=argparse.BooleanOptionalAction, default=False)

group3 = generic_parser.add_argument_group('Misc/Logging arguments')
group3.add_argument("--freeze-sigma", action=argparse.BooleanOptionalAction, default=False)
group3.add_argument("--mc-eval-samples", type=int, default=1)
group3.add_argument("--mc-train-samples", type=int, default=1)
group3.add_argument(
    "--loglevel",
    choices=["debug", "info", "error", "warning", "critical"],
    default="debug",
)

def remove_argument(parser, arg):
    for action in parser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            parser._remove_action(action)
            break

    for action in parser._action_groups:
        for group_action in action._group_actions:
            opts = group_action.option_strings
            if (opts and opts[0] == arg) or group_action.dest == arg:
                action._group_actions.remove(group_action)
                return