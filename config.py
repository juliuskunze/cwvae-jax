from argparse import ArgumentParser
from dataclasses import dataclass, asdict, field, fields, replace
from pathlib import Path
from typing import Optional, List

import numpy as np
import yaml


def must_be(value): return field(default=value, metadata=dict(choices=[value]))


@dataclass
class Config:
    config: str  # Path to config yaml file
    datadir: str  # Path to root data directory
    logdir: str  # Path to root log directory (eval: dir containing model checkpoint with config in the parent dir)

    # MODEL
    levels: int = 3  # Number of levels in the hierarchy
    tmp_abs_factor: int = 6  # Temporal abstraction factor used at each level
    dec_stddev: float = 1.  # Standard deviation of the decoder distribution
    enc_dense_layers: int = 3  # Number of dense hidden layers at each level
    enc_dense_embed_size: int = 1000  # Size of dense hidden embeddings
    cell_stoch_size: int = 20
    cell_deter_size: int = 200
    cell_embed_size: int = 200
    cell_min_stddev: float = .0001  # Minimum standard deviation of prior and posterior distributions
    use_obs: Optional[str] = None  # String of T/Fs per level, e.g. TTF to skip obs at the top level
    channels_mult: int = 1  # Multiplier for the number of channels in the conv encoder
    filters: int = 32  # Base number of channels in the convolutions

    # DATASET
    dataset: str = field(
        default="mmnist", metadata=dict(choices=["mmnist", "minerl", "mazes"]))
    seq_len: int = 100  # Length of training sequences
    eval_seq_len: int = 1000  # Total length of evaluation sequences
    channels: int = 1  # Number of channels in the output video

    # TRAINING
    lr: float = .0003
    batch_size: int = 50
    num_epochs: int = 300
    clip_grad_norm_by: float = 10000
    seed: int = np.random.randint(np.iinfo(np.int32).max)

    # SUMMARIES
    open_loop_ctx: int = 36  # Number of context frames for open loop prediction
    save_gifs: bool = True
    save_scalars_every: int = 1000
    save_gifs_every: int = 1000
    save_model_every: int = 1000
    save_named_model_every: int = 5000

    # EVALUATION
    num_examples: int = 100  # Number of examples to evaluate on
    num_samples: int = 1  # Samples to generate per example
    no_save_grid: bool = False  # To prevent saving grids of images

    # NOT IMPLEMENTED, must have the default value:
    cell_type: str = must_be("RSSMCell")
    cell_mean_only: str = must_be('false')
    cell_reset_state: str = must_be('false')
    beta: Optional[float] = must_be(None)
    free_nats: Optional[float] = must_be(None)
    kl_grad_post_perc: Optional[float] = must_be(None)
    num_val_batches: int = must_be(1)

    def config_file(self, eval):
        return Path(self.logdir).parent / "config.yml" if eval else Path(self.config)

    @property
    def _run_name(self):
        return f"{self.dataset}_cwvae_{self.cell_type.lower()}" \
               f"_{self.levels}l_f{self.tmp_abs_factor}_decsd{self.dec_stddev}" \
               f"_enchl{self.enc_dense_layers}_ences{self.enc_dense_embed_size}" \
               f"_edchnlmult{self.channels_mult}_ss{self.cell_stoch_size}" \
               f"_ds{self.cell_deter_size}_es{self.cell_embed_size}" \
               f"_seq{self.seq_len}_lr{self.lr}_bs{self.batch_size}"

    @property
    def exp_rootdir(self):
        return Path(self.logdir) / self.dataset / self._run_name

    def save(self):
        self.exp_rootdir.mkdir(parents=True, exist_ok=True)
        with (self.exp_rootdir / "config.yml").open("w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @property
    def total_filters(self):
        return self.filters * self.channels_mult

    @property
    def use_observations(self) -> List[bool]:
        if self.use_obs is None:
            return [True] * self.levels
        assert len(self.use_obs) == self.levels
        return [dict(T=True, F=False)[c] for c in self.use_obs.upper()]

    @property
    def _dataset_name(self):
        return dict(minerl="minerl_navigate",
                    mmnist="moving_mnist_2digit",
                    mazes="gqn_mazes")[self.dataset]

    def load_dataset(self, eval=False):
        import tensorflow as tf
        import tensorflow_datasets as tfds

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        if self.dataset == "minerl":
            import minerl_navigate
        elif self.dataset == "mmnist":
            import datasets.moving_mnist
        elif self.dataset == "mazes":
            import datasets.gqn_mazes
        d = tfds.load(self._dataset_name,
                      data_dir=self.datadir, shuffle_files=not eval)
        d = d["test" if eval else "train"]
        d = d.map(lambda vid: tf.cast(vid["video"], tf.float32) / 255.0)
        seq_len = self.eval_seq_len if eval else self.seq_len
        if seq_len:
            def split_to_seq_len(seq):
                usable_len = tf.shape(seq)[0] - (tf.shape(seq)[0] % seq_len)
                seq = tf.reshape(seq[:usable_len], tf.concat(
                    [[usable_len // seq_len, seq_len], tf.shape(seq)[1:]], -1))
                return tf.data.Dataset.from_tensor_slices(seq)

            d = d.flat_map(split_to_seq_len)
        d = d.prefetch(tf.data.experimental.AUTOTUNE)
        if not eval:
            d = d.repeat(self.num_epochs).shuffle(10 * self.batch_size)
        d = d.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return tfds.as_numpy(d)


def parse_config(eval=False):
    p = ArgumentParser()
    for f in fields(Config):
        kwargs = (
            dict(action='store_true') if f.type is bool and not f.default else
            dict(default=f.default, type=f.type))
        p.add_argument(f'--{f.name}', **kwargs, **f.metadata)
    c = Config(**vars(p.parse_args()))
    p.set_defaults(**yaml.full_load(c.config_file(eval).read_text()))
    return replace(c, **vars(p.parse_args()))
