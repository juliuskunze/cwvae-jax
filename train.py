from functools import partial
from itertools import islice

import jax
import numpy as np
import wandb
from flax.optim import Adam
from flax.training import checkpoints
from jax import jit, value_and_grad, numpy as jnp, random

from config import Config, parse_config
from cwvae import Model
from tools import log, video

if __name__ == "__main__":
    c = parse_config()
    with wandb.init(config=c):
        c = Config(**wandb.config)
        c.save()
        train_batches = c.load_dataset()
        val_batch = next(iter(c.load_dataset(eval=True)))
        model = Model(c)


        @jit
        def train_step(state, rng, obs):
            rng, sample_rng = random.split(rng)
            loss_fn = partial(model.apply, obs=obs,
                              rngs=dict(sample=sample_rng))
            grad_fn = value_and_grad(loss_fn, has_aux=True)
            (_, metrics), grad = grad_fn(state.target)
            grad_norm = jnp.linalg.norm(
                jax.tree_leaves(jax.tree_map(jnp.linalg.norm, grad)))
            if c.clip_grad_norm_by:
                # Clipping gradients by global norm
                scale = jnp.minimum(c.clip_grad_norm_by / grad_norm, 1)
                grad = jax.tree_map(lambda x: scale * x, grad)
            metrics['grad_norm'] = grad_norm
            return state.apply_gradient(grad), rng, metrics


        @jit
        def get_metrics(state, rng, obs):
            _, metrics = model.apply(state.target, obs=obs,
                                     rngs=dict(sample=rng))
            return metrics


        @jit
        def get_video(state, rng, obs):
            return video(pred=model.apply(
                state.target, obs=obs, rngs=dict(sample=rng),
                method=model.open_loop_unroll), target=obs[:, c.open_loop_ctx:])


        rng, video_rng, params_rng, sample_rng = random.split(
            random.PRNGKey(c.seed), 4)
        params = model.init(dict(params=params_rng, sample=sample_rng),
                            next(iter(train_batches)))
        state = Adam(learning_rate=c.lr, eps=1e-4).create(params)
        state, rng = checkpoints.restore_checkpoint(c.exp_rootdir, (state, rng))
        if state.state.step:
            print(f"Restored model from {c.exp_rootdir}")
            print(f"Will start training from step {state.state.step}")
            train_batches = islice(train_batches, state.state.step, None)

        print("Training.")
        for train_batch in train_batches:
            state, rng, metrics = train_step(state, rng, train_batch)
            step = state.state.step
            print(f"batch {step}: loss {metrics['loss']:.1f}")

            if step % c.save_scalars_every == 0:
                log(metrics, step, 'train/')
                log(get_metrics(state, rng, val_batch), step, 'val/')

            if c.save_gifs and step % c.save_gifs_every == 0:
                v = np.array(get_video(state, video_rng, train_batch))
                log(dict(pred_video=wandb.Video(v, fps=10)), step, 'train/')
                v = np.array(get_video(state, video_rng, val_batch))
                log(dict(pred_video=wandb.Video(v, fps=10)), step, 'val/')

            if step % c.save_model_every == 0:
                checkpoints.save_checkpoint(c.exp_rootdir, (state, rng), step)

            if c.save_named_model_every and step % c.save_named_model_every == 0:
                checkpoints.save_checkpoint(c.exp_rootdir / f"model_{step}",
                                            (state, rng), step)

        print("Training complete.")
