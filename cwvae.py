from flax import linen as nn
from jax import numpy as jnp
from jax.util import safe_zip as zip
from tensorflow_probability.substrates.jax import distributions as tfd

from cells import RSSMCell
from cnns import Encoder, Decoder
from config import Config


class CWVAE(nn.Module):
    c: Config

    @nn.compact
    def __call__(self, inputs, use_observations=None, initial_state=None):
        """
        Used to unroll a list of recurrent cells.

        Arguments:
            inputs : list of encoded observations
                Number of timesteps at every level in 'inputs' is the number of steps to be unrolled.
            use_observations : None or list[bool]
            initial_state : list of cell states
        """
        if use_observations is None:
            use_observations = self.c.levels * [True]
        if initial_state is None:
            initial_state = self.c.levels * [None]

        cells = [RSSMCell(self.c) for _ in range(self.c.levels)]

        priors = []
        posteriors = []
        last_states = []
        is_top_level = True
        for level, (cell, use_obs, obs_inputs, initial) in reversed(list(
                enumerate(
                    zip(cells, use_observations, inputs, initial_state)))):

            print(f"Input shape in CWVAE level {level}: {obs_inputs.shape}")

            if is_top_level:
                # Feeding in zeros as context to the top level:
                context = jnp.zeros(obs_inputs.shape[:2] + (cell.state_size["output"],))
                is_top_level = False
            else:
                # Tiling context from previous layer in time by tmp_abs_factor:
                context = jnp.expand_dims(context, axis=2)
                context = jnp.tile(context, [1, 1, self.c.tmp_abs_factor]
                                   + (len(context.shape) - 3) * [1])
                s = context.shape
                context = context.reshape((s[0], s[1] * s[2]) + s[3:])
                # Pruning timesteps to match inputs:
                context = context[:, :obs_inputs.shape[1]]

            # Unroll of RNN cell.
            scan = nn.scan(
                lambda c, state, xs: c(state, xs, use_obs=use_obs),
                variable_broadcast='params',
                split_rngs=dict(params=False, sample=True),
                in_axes=1, out_axes=1)

            initial = cell.zero_state(obs_inputs.shape[0]
                                      ) if initial is None else initial
            last_state, (prior, posterior) = scan(cell, initial,
                                                  (obs_inputs, context))
            context = posterior["output"]

            last_states.insert(0, last_state)
            priors.insert(0, prior)
            posteriors.insert(0, posterior)
        return last_states, priors, posteriors

    def open_loop_unroll(self, inputs):
        assert self.c.open_loop_ctx % (
                self.c.tmp_abs_factor ** (self.c.levels - 1)) == 0, \
            f"Incompatible open-loop context length {self.open_loop_ctx} and " \
            f"temporal abstraction factor {self.tmp_abs_factor} for levels {self.levels}."
        ctx_lens = [self.c.open_loop_ctx // self.c.tmp_abs_factor ** level
                    for level in range(self.c.levels)]
        pre_inputs, post_inputs = zip(*[
            (input[:, :ctx_len], jnp.zeros_like(input[:, ctx_len:]))
            for input, ctx_len in zip(inputs, ctx_lens)])

        last_states, _, _ = self(
            pre_inputs, use_observations=self.c.use_observations)
        _, predictions, _ = self(
            post_inputs, use_observations=self.c.levels * [False],
            initial_state=last_states)
        return predictions


class Model(nn.Module):
    c: Config

    def setup(self):
        self.encoder = Encoder(self.c)
        self.model = CWVAE(self.c)
        self.decoder = Decoder(self.c)

    def decode(self, predictions):
        bottom_layer_output = predictions[0]['output']
        return self.decoder(bottom_layer_output)

    def __call__(self, obs):
        assert obs.shape[-3:] == (64, 64, self.c.channels)
        _, priors, posteriors = self.model(self.encoder(obs))
        output = tfd.Independent(
            tfd.Normal(self.decode(posteriors), self.c.dec_stddev))
        priors = [tfd.Independent(tfd.Normal(d["mean"], d["stddev"]))
                  for d in priors]
        posteriors = [tfd.Independent(tfd.Normal(d["mean"], d["stddev"]))
                      for d in posteriors]

        nll_term = -jnp.mean(output.log_prob(obs), 0)
        kls = [jnp.mean(posterior.kl_divergence(prior), 0)
               for prior, posterior in zip(priors, posteriors)]
        kl_term = sum(kls)
        metrics = dict(loss=nll_term + kl_term,
                       kl_term=kl_term, nll_term=nll_term)
        for lvl, (prior, posterior, kl) in enumerate(
                zip(priors, posteriors, kls)):
            metrics.update({
                f"avg_kl_prior_posterior__level_{lvl}": kl,
                f"avg_entropy_prior__level_{lvl}": jnp.mean(prior.entropy(), 0),
                f"avg_entropy_posterior__level_{lvl}": jnp.mean(
                    posterior.entropy(), 0)
            })

        per_timestep_metrics = {k: v / obs.shape[1] for k, v in metrics.items()}
        return per_timestep_metrics['loss'], per_timestep_metrics

    def open_loop_unroll(self, obs):
        return self.decode(self.model.open_loop_unroll(self.encoder(obs)))
