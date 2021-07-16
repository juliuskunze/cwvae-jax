import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import wandb
from jax import numpy as jnp
from jax.interpreters.xla import DeviceArray
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric


def log(kwargs, step=None, prefix=''):
    wandb.log({prefix + k: np.array(x) if isinstance(x, DeviceArray) else x
               for k, x in kwargs.items()}, step=step)


def _to_padded_strip(images):
    if len(images.shape) <= 3:
        images = np.expand_dims(images, -1)
    c_dim = images.shape[-1]
    x_dim = images.shape[-3]
    y_dim = images.shape[-2]
    padding = 1
    result = np.zeros((x_dim, y_dim * images.shape[0] +
                       padding * (images.shape[0] - 1), c_dim), dtype=np.uint8)
    for i in range(images.shape[0]):
        result[:, i * y_dim + i * padding:
                  (i + 1) * y_dim + i * padding] = images[i]
    if result.shape[-1] == 1:
        result = np.reshape(result, result.shape[:2])
    return result


def save_as_grid(images, save_dir, filename, strip_width=50):
    # Creating a grid of images.
    # images shape: (T, ...)
    results = []
    if images.shape[0] < strip_width:
        results.append(_to_padded_strip(images))
    else:
        for i in range(0, images.shape[0], strip_width):
            if i + strip_width < images.shape[0]:
                results.append(_to_padded_strip(images[i: i + strip_width]))
    grid = np.concatenate(results, 0)
    imageio.imwrite(os.path.join(save_dir, filename), grid)
    print(f"Written grid file {os.path.join(save_dir, filename)}")


def compute_metrics(gt, pred):
    gt = np.transpose(gt, [0, 1, 4, 2, 3])
    pred = np.transpose(pred, [0, 1, 4, 2, 3])
    bs = gt.shape[0]
    T = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            for c in range(gt[i][t].shape[0]):
                ssim[i, t] += ssim_metric(gt[i][t][c], pred[i][t][c])
                psnr[i, t] += psnr_metric(gt[i][t][c], pred[i][t][c])
            ssim[i, t] /= gt[i][t].shape[0]
            psnr[i, t] /= gt[i][t].shape[0]

    return ssim, psnr


def plot_metrics(metrics, logdir, name):
    mean_metric = np.squeeze(np.mean(metrics, 0))
    stddev_metric = np.squeeze(np.std(metrics, 0))
    np.savez(os.path.join(logdir, f"{name}_mean.npz"), mean_metric)
    np.savez(os.path.join(logdir, f"{name}_stddev.npz"), stddev_metric)

    plt.figure()
    axes = plt.gca()
    axes.yaxis.grid(True)
    plt.plot(mean_metric, color="blue")
    axes.fill_between(
        np.arange(0, mean_metric.shape[0]),
        mean_metric - stddev_metric,
        mean_metric + stddev_metric,
        color="blue",
        alpha=0.4,
    )
    plt.savefig(os.path.join(logdir, f"{name}_range.png"))


def video(pred, target, max_batch=8, clip_by=(0., 1.)):
    # Inputs are expected to be (batch, time, height, width, channels).
    image = jnp.clip(pred[:max_batch], clip_by[0], clip_by[1])

    image = (image - clip_by[0]) / (clip_by[1] - clip_by[0])
    target = (target - clip_by[0]) / (clip_by[1] - clip_by[0])

    target = target[:max_batch]
    error = ((image - target) + 1) / 2
    # Concat ground truth, prediction, and error vertically.
    frames = jnp.concatenate([target, image, error], 2)
    # Concat batch entries horizontally and pull channels forward.
    frames = frames.transpose((1, 4, 2, 0, 3))
    frames = frames.reshape(frames.shape[:3] + (-1,))
    return (255 * frames).astype(np.uint8)
