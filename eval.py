from datetime import datetime
from pathlib import Path

import numpy as np
from flax.optim import Adam
from flax.training import checkpoints
from jax import random, jit

import tools
from config import parse_config
from cwvae import Model

if __name__ == "__main__":
    c = parse_config(eval=True)
    c.batch_size = 1
    model_dir = Path(c.logdir)
    now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    eval_logdir = model_dir.parent / f"eval_{now}"
    eval_logdir.mkdir(exist_ok=True)
    val_batches = c.load_dataset(eval=True)
    model = Model(c)

    print(f"Restoring model from {c.logdir}")
    rng, params_rng, sample_rng = random.split(random.PRNGKey(c.seed), 3)
    params = model.init(dict(params=params_rng, sample=sample_rng),
                        next(iter(val_batches)))
    state = Adam(learning_rate=c.lr).create(params)
    state, _ = checkpoints.restore_checkpoint(model_dir, (state, rng))
    assert state.state.step
    print(f"Evaluating model at step {state.state.step}")


    @jit
    def open_loop_preds(obs, sample_rng):
        return model.apply(state.target, obs, rngs=dict(sample=sample_rng),
                           method=model.open_loop_unroll)


    # Evaluating.
    ssim_all = []
    psnr_all = []
    for i_ex, val_batch in zip(range(c.num_examples), val_batches):
        gts = np.tile(val_batch, [c.num_samples, 1, 1, 1, 1])

        rng, sample_rng = random.split(rng, 2)
        preds = np.array(open_loop_preds(gts, sample_rng))
        # Computing metrics.
        ssim, psnr = tools.compute_metrics(gts[:, c.open_loop_ctx:], preds)

        # Getting arrays save-ready
        gts = np.uint8(np.clip(gts, 0, 1) * 255)
        preds = np.uint8(np.clip(preds, 0, 1) * 255)

        # Finding the order within samples wrt avg metric across time.
        order_ssim = np.argsort(np.mean(ssim, -1))
        order_psnr = np.argsort(np.mean(psnr, -1))

        # Setting aside the best metrics among all samples for plotting.
        ssim_all.append(np.expand_dims(ssim[order_ssim[-1]], 0))
        psnr_all.append(np.expand_dims(psnr[order_psnr[-1]], 0))

        # Storing gt for prediction and the context.
        path = eval_logdir / f"sample{i_ex}_gt"
        path.mkdir(exist_ok=True)
        np.savez(path / "gt_ctx.npz", gts[0, : c.open_loop_ctx])
        np.savez(path / "gt_pred.npz", gts[0, c.open_loop_ctx:])
        if not c.no_save_grid:
            tools.save_as_grid(gts[0, : c.open_loop_ctx], path, "gt_ctx.png")
            tools.save_as_grid(gts[0, c.open_loop_ctx:], path, "gt_pred.png")

        # Storing best and random samples.
        path = eval_logdir / f"sample{i_ex}"
        path.mkdir(exist_ok=True)
        np.savez(path / "random_sample_1.npz", preds[0])
        if c.num_samples > 1:
            np.savez(path / "best_ssim_sample.npz", preds[order_ssim[-1]])
            np.savez(path / "best_psnr_sample.npz", preds[order_psnr[-1]])
            np.savez(path / "random_sample_2.npz", preds[1])
        if not c.no_save_grid:
            tools.save_as_grid(preds[0], path, "random_sample_1.png")
            if c.num_samples > 1:
                tools.save_as_grid(
                    preds[order_ssim[-1]], path, "best_ssim_sample.png")
                tools.save_as_grid(
                    preds[order_psnr[-1]], path, "best_psnr_sample.png")
                tools.save_as_grid(preds[1], path, "random_sample_2.png")

    # Plotting.
    tools.plot_metrics(ssim_all, eval_logdir, "ssim")
    tools.plot_metrics(psnr_all, eval_logdir, "psnr")
