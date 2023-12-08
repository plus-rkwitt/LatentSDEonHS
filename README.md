# Latent SDEs on Homogeneous Spaces

<p align="center">
  <img width="300" height="300" src="./assets/spherepaths.gif">
</p>

This repository contains the **reference implementation** of 

Zeng, S. and Graf, F. and Kwitt, R.    
**Latent SDEs on Homogeneous Spaces**    
*NeurIPS 2023*

Please cite as:

```bibtex
@inproceedings{Zeng23a,
    author    = {Zeng, Sebastian and Graf, Florian and Kwitt, Roland},
    title     = {Latent SDEs on Homogeneous Spaces},
    booktitle = {NeurIPS},
    year      = {2023}}
```
A preprint is available on [arXiv](https://arxiv.org/abs/2306.16248). In the following, we refer to the 
paper as [**Zeng23a**].

# Overview

- [Setup](#setup)
- [Experiments](#experiments)
  - [Rotating MNIST](#rotating-mnist)
  - [Pendulum (Angle) Regression](#pendulum-angle-regression)
  - [Pendulum Interpolation](#pendulum-interpolation)
  - [Human Activity Classification](#human-activity-classification)
  - [PhysioNet 2012 Interpolation](#physionet-2012-interpolation)
  - [Irregular Sine (Toy) Experiment](#irregular-sine-toy-experiment)
- [Notebooks](#notebooks)

# Setup

### Example setup

The following example (1) sets up Anaconda Python (2023.09) in `/tmp/anaconda3`, (2) creates and activates a virtual environment `pytorch211`, (3) installs PyTorch 2.1.1 (according to the [PyTorch](https://pytorch.org/) installation instructions as of Dec. 2023) and (4) installs all dependencies, i.e., `einops, scipy, scikit-learn` and `gdown`.

```bash
cd /tmp/
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
chmod +x Anaconda3-2023.09-0-Linux-x86_64.sh
./Anaconda3-2023.09-0-Linux-x86_64.sh -p /tmp/anaconda3 
source source /tmp/anaconda3/bin/activate
conda create --name pytorch211 python=3.11
conda activate pytorch211
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip3 install einops scipy scikit-learn gdown
```

Next, clone the repository, get the implementation of the power spherical distribution and set it up 
as listed:

```bash
cd /tmp/
git clone https://github.com/plus-rkwitt/LatentSDEonHS.git
git clone https://github.com/nicola-decao/power_spherical.git
cd LatentSDEonHS
mv ../power_spherical/power_spherical core/
```

If you can run (within the activated `pytorch211` environment)
```bash
python rotating_mnist.py --help 
```
without any errors you are good to go.

### Test System

This code has been mainly tested on an Ubuntu Linux 22.04 with an Nvidia GTX 3090 GPU, CUDA 12.1 (driver version 530.30.02 ) and PyTorch v2.1.1. We have also tested the code running PyTorch 1.13 on the same system (CUDA 11.8).

# Experiments

The following sections lists the settings we used to run the 
experiments from the manuscript. In the configurations listed
below, experiments are run on the first GPU in your system 
(`cuda:0`).

The `logs` directory will hold (if `--enable-file-logging` is set and a valid directory is provided with `--log-dir`) **two** files per experiment, tracking (experiment-specific) performance measures, i.e., a JSON  (`.json`) file as well as an exact replica of the console output (the `.txt` file). Each file is identified by a unique experiment identifier.

## Rotating MNIST

```bash
python rotating_mnist.py \
    --data-dir data_dir \
    --enable-file-logging  \
    --log-dir logs \
    --no-enable-checkpointing  \
    --checkpoint-dir None \
    --checkpoint-at  \
    --batch-size 32 \
    --lr 0.001 \
    --n-epochs 990 \
    --kl0-weight 0.0001 \
    --klp-weight 0.0001 \
    --pxz-weight 1.0 \
    --seed -1 \
    --restart 30 \
    --device cuda:0 \
    --z-dim 16 \
    --h-dim 32 \
    --n-deg 6 \
    --no-learnable-prior \
    --no-freeze-sigma  \
    --mc-eval-samples 1 \
    --mc-train-samples 1 \
    --loglevel debug \
    --n-filters 8
```

Below are the results of **two** runs with different random seeds:

| Run  | MSE (on left-out image) $\times 10^{-3}$ |
|---|---|
| `1` | `11.05` | 
| `2` | `11.25` | 

In [**Zeng23a**], we report an MSE of `11.8 +/- 0.25`.


## Pendulum (Angle) Regression
```bash
python pendulum_regression.py \
    --data-dir data_dir \
    --enable-file-logging  \
    --log-dir logs \
    --no-enable-checkpointing  \
    --checkpoint-dir None \
    --checkpoint-at  \
    --batch-size 64 \
    --lr 0.001 \
    --n-epochs 990 \
    --kl0-weight 1e-05 \
    --klp-weight 1e-06 \
    --pxz-weight 1.0 \
    --seed -1 \
    --restart 30 \
    --device cuda:0 \
    --z-dim 16 \
    --h-dim 32 \
    --n-deg 6 \
    --no-learnable-prior  \
    --no-freeze-sigma  \
    --mc-eval-samples 1 \
    --mc-train-samples 1 \
    --loglevel debug \
    --aux-weight 10.0 \
    --aux-hidden-dim 32 \
    --use-atanh
```

Below are the results of **two** runs with different random seeds:

| Run  | MSE $\times 10^{-3}$ |
|---|---|
| `1` | `4.13` | 
| `2` | `3.81` | 

In [**Zeng23a**], we report an average MSE of `4.23 +/- 0.5`.


## Pendulum Interpolation

```bash
python pendulum_interpolation.py \
    --data-dir data_dir \
    --enable-file-logging  \
    --log-dir logs \
    --no-enable-checkpointing  \
    --checkpoint-dir None \
    --checkpoint-at  \
    --batch-size 64 \
    --lr 0.001 \
    --n-epochs 990 \
    --kl0-weight 1e-05 \
    --klp-weight 1e-06 \
    --pxz-weight 1.0 \
    --seed -1 \
    --restart 30 \
    --device cuda:0 \
    --z-dim 16 \
    --h-dim 32 \
    --n-deg 6 \
    --no-learnable-prior  \
    --no-freeze-sigma  \
    --mc-eval-samples 1 \
    --mc-train-samples 1 \
    --loglevel debug \
    --use-atanh
```

Below are the results of **two** runs with different random seeds:]

| Run  | MSE $\times 10^{-3}$ |
|---|---|
| `1` | `8.06` | 
| `2` | `8.26` | 

In [**Zeng23a**], we report an MSE of `8.02 +/- 0.10`.

## Human Activity Classification

For the human activity recognition experiment, just run the `activity_classification.py` script with default
arguments. **Note**: different to the manuscript, we do actually reconstruct the input here at the available 
timepoints (and at the available coordinates). This is not necessary, but in that manner, all experiments are
fully consistent in their setup.

```bash
python activity_classification.py \
    --data-dir data_dir \
    --enable-file-logging  \
    --log-dir logs \
    --no-enable-checkpointing  \
    --checkpoint-dir None \
    --checkpoint-at  \
    --batch-size 64 \
    --lr 0.001 \
    --n-epochs 990 \
    --kl0-weight 0.0001 \
    --klp-weight 0.0001 \
    --pxz-weight 1.0 \
    --seed -1 \
    --restart 30 \
    --device cuda:0 \
    --z-dim 16 \
    --h-dim 128 \
    --n-deg 4 \
    --no-learnable-prior  \
    --freeze-sigma  \
    --mc-eval-samples 1 \
    --mc-train-samples 1 \
    --loglevel debug \
    --aux-weight 10.0 \
    --aux-hidden-dim 32 \
    --use-atanh
```

Below are the results of **two** runs with different random seeds:

| Run  | Test Accuracy (\%) |
|---|---|
| 1 | `90.92` | 
| 2 | `90.70` | 

In [**Zeng23a**], we report an accuracy of `90.7 +/- 0.3`.

## PhysioNet 2012 Interpolation

The PhysioNet 2012 interpolation experiments can be found in Section 4.2. of [**Zeng23a**].

### Quantization of 6 minutes

```bash
python physionet_interpolation.py \
    --data-dir data_dir \
    --enable-file-logging  \
    --log-dir logs \
    --no-enable-checkpointing  \
    --checkpoint-dir None \
    --checkpoint-at  \
    --batch-size 64 \
    --lr 0.001 \
    --n-epochs 990 \
    --kl0-weight 0.0001 \
    --klp-weight 0.0001 \
    --pxz-weight 1.0 \
    --seed -1 \
    --restart 30 \
    --device cuda:1 \
    --z-dim 16 \
    --h-dim 64 \
    --n-deg 6 \
    --no-learnable-prior  \
    --freeze-sigma  \
    --mc-eval-samples 1 \
    --mc-train-samples 1 \
    --loglevel debug \
    --n-dec-layers 2 \
    --dec-hidden-dim 100 \
    --no-use-atanh  \
    --sample-tp 0.5 \
    --quantization 0.1
```
Below are the results of **two** runs with different random seeds at `--sample-tp 0.5` which means that 50\% of timepoints
where there are actual measurements are taken as model input:

| Run  | MSE (on remaining measurements) $\times 10^{-3}$ |
|---|---|
| 1 | `3.19` | 
| 2 | `3.12` | 

At that sampling rate and quantization (6 minutes equals `--quantization 0.1`), we report `3.25 +/- 0.02` in [**Zeng23a**].

## Irregular Sine (Toy) Experiment

`irregular_sine_interpolation.py` implements a Latent SDE model for the *irregular sine* data from [torchsde](https://github.com/google-research/torchsde/).
You can run the code with

```bash
python irregular_sine_interpolation.py \
    --enable-file-logging  \
    --log-dir logs \
    --enable-checkpointing  \
    --checkpoint-dir checkpoints \
    --checkpoint-at 90 390 990 2190 3990  \
    --lr 0.001 \
    --n-epochs 3990 \
    --kl0-weight 0.001 \
    --klp-weight 0.01 \
    --pxz-weight 1.0 \
    --seed -1 \
    --restart 30 \
    --device cuda:0 \
    --z-dim 3 \
    --h-dim 3 \
    --n-deg 6 \
    --no-learnable-prior  \
    --freeze-sigma  \
    --mc-eval-samples 1 \
    --mc-train-samples 1 \
    --loglevel debug
```

Note that the batch-size, by construction of that experiment, equals one; hence, the  `--n-epochs` corresponds to the number of update
steps. The introductory figure for this README shows (in blue) the progression of several latent paths on the 2-sphere, (as `--z-dim 3`) 
across the time interval [0,1].

# Notebooks

Several notebooks are available (in the `notebooks` subfolder) to analyze and visualize the results, aside from the command-line tracking of performance measures.

