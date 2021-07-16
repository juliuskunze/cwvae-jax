# Clockwork VAEs in JAX/Flax

Implementation of experiments in the paper [Clockwork Variational Autoencoders](https://arxiv.org/pdf/2102.09532.pdf) ([project website](http://danijar.com/cwvae)) using [JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax), ported from the [official TensorFlow implementation](https://github.com/vaibhavsaxena11/cwvae).

Running on a single TPU v3, training is **10x faster** than reported in the paper (60h -> 6h on `minerl`).

## Method

<img src="https://danijar.com/asset/cwvae/header.gif">

Clockwork VAEs are deep generative model that learn long-term dependencies in video by leveraging hierarchies of representations that progress at different clock speeds. In contrast to prior video prediction methods that typically focus on predicting sharp but short sequences in the future, Clockwork VAEs can accurately predict high-level content, such as object positions and identities, for 1000 frames.

Clockwork VAEs build upon the [Recurrent State Space Model (RSSM)](https://arxiv.org/pdf/1811.04551.pdf), so each state contains a deterministic component for long-term memory and a stochastic component for sampling diverse plausible futures. Clockwork VAEs are trained end-to-end to optimize the evidence lower bound (ELBO) that consists of a reconstruction term for each image and a KL regularizer for each stochastic variable in the model.

## Instructions

This repository contains the code for training the Clockwork VAE model on the datasets `minerl`, `mazes`, and `mmnist`.

The datasets will automatically be downloaded into the `--datadir` directory.

```sh
python3 train.py --logdir /path/to/logdir --datadir /path/to/datasets --config configs/<dataset>.yml 
```

The evaluation script writes open-loop video predictions in both PNG and NPZ format and plots of PSNR and SSIM to the data directory.

```sh
python3 eval.py --logdir /path/to/logdir
```

## Known differences from the original

- Flax' default kernel initializer, layer precision and GRU implementation (avoiding redundant biases) are used.
- For some configuration parameters, only the defaults are implemented.
- Training metrics and videos are logged with `wandb`.
- The base configuration is in `config.py`.

Added features:

- This implementation runs on TPU out-of-the-box.
- Apart from the config file, configuration can be done via command line and `wandb`.
- Matching the `seed` of a previous run will exactly repeat it.

## Things to watch out for

Replication of paper results for the `mazes` dataset has not been confirmed yet.

Getting evaluation metrics is a memory bottleneck during training, due to the large `eval_seq_len`. 
If you run out of device memory, consider lowering it during training, for example to 100. 
Remember to pass in the original value to `eval.py` to get unchanged results.

## Acknowledgements

Thanks to [Vaibhav Saxena](https://github.com/vaibhavsaxena11) and [Danijar Hafner](https://danijar.com) for helpful discussions and to [Jamie Townsend](https://github.com/j-towns) for reviewing code.