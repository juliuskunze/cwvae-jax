from flax import linen as nn
from flax.linen import GRUCell
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from config import Config


class RSSMPrior(nn.Module):
    c: Config

    @nn.compact
    def __call__(self, prev_state, context):
        inputs = jnp.concatenate([prev_state["sample"], context], -1)
        hl = nn.relu(nn.Dense(self.c.cell_embed_size)(inputs))
        det_state, det_out = GRUCell()(prev_state["det_state"], hl)
        hl = nn.relu(nn.Dense(self.c.cell_embed_size)(det_out))
        mean = nn.Dense(self.c.cell_stoch_size)(hl)
        stddev = nn.softplus(
            nn.Dense(self.c.cell_stoch_size)(hl + .54)) + self.c.cell_min_stddev
        dist = tfd.MultivariateNormalDiag(mean, stddev)
        sample = dist.sample(seed=self.make_rng('sample'))
        return dict(mean=mean, stddev=stddev, sample=sample,
                    det_out=det_out, det_state=det_state,
                    output=jnp.concatenate([sample, det_out], -1))


class RSSMPosterior(nn.Module):
    c: Config

    @nn.compact
    def __call__(self, prior, obs_inputs):
        inputs = jnp.concatenate([prior["det_out"], obs_inputs], -1)
        hl = nn.relu(nn.Dense(self.c.cell_embed_size)(inputs))
        hl = nn.relu(nn.Dense(self.c.cell_embed_size)(hl))
        mean = nn.Dense(self.c.cell_stoch_size)(hl)
        stddev = nn.softplus(
            nn.Dense(self.c.cell_stoch_size)(hl + .54)) + self.c.cell_min_stddev
        dist = tfd.MultivariateNormalDiag(mean, stddev)
        sample = dist.sample(seed=self.make_rng('sample'))
        return dict(mean=mean, stddev=stddev, sample=sample,
                    det_out=prior["det_out"], det_state=prior["det_state"],
                    output=jnp.concatenate([sample, prior["det_out"]], -1))


class RSSMCell(nn.Module):
    c: Config

    @property
    def state_size(self):
        return dict(
            mean=self.c.cell_stoch_size, stddev=self.c.cell_stoch_size,
            sample=self.c.cell_stoch_size, det_out=self.c.cell_deter_size,
            det_state=self.c.cell_deter_size,
            output=self.c.cell_stoch_size + self.c.cell_deter_size)

    def zero_state(self, batch_size, dtype=jnp.float32):
        return {k: jnp.zeros((batch_size, v), dtype=dtype)
                for k, v in self.state_size.items()}

    @nn.compact
    def __call__(self, state, inputs, use_obs):
        obs_input, context = inputs
        prior = RSSMPrior(self.c)(state, context)
        posterior = RSSMPosterior(self.c)(prior,
                                          obs_input) if use_obs else prior
        return posterior, (prior, posterior)
