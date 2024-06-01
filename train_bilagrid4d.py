import glob
import logging
import os
import shutil
import sys

import numpy as np
import random

import time

from absl import app
import gin
from internal import configs
from internal import datasets
from internal import image
from internal import models
from internal import train_utils
from internal import utils
from internal import vis
from internal import checkpoints
import torch
import accelerate
import tensorboardX
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from torch.utils._pytree import tree_map


TIME_PRECISION = 1000  # Internally represent integer times in milliseconds.



def main(unused_argv):
    config = configs.load_config()
    config.exp_path = os.path.join("exp", config.exp_name)
    config.checkpoint_dir = os.path.join(config.exp_path, 'checkpoints')
    if config.ft_reload_ckpt:
        config.checkpoint_dir = config.ft_reload_ckpt
    config.ft_path = os.path.join(config.exp_path, 'ft', config.ft_name)
    config.ft_checkpoint_dir = os.path.join(config.ft_path, 'checkpoints')
    config.test_vis_dir = os.path.join(config.ft_path, 'ft_vis')
    utils.makedirs(config.exp_path)
    utils.makedirs(config.test_vis_dir)
    utils.makedirs(config.ft_path)
    with utils.open_file(os.path.join(config.ft_path, 'config_ft.gin'), 'w') as f:
        f.write(gin.config_str())

    # accelerator for DDP
    accelerator = accelerate.Accelerator()

    # setup logger
    logging.basicConfig(
        format="%(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(os.path.join(config.ft_path, 'log_ft.txt'))],
        level=logging.INFO,
    )
    sys.excepthook = utils.handle_exception
    logger = accelerate.logging.get_logger(__name__)
    logger.info(config)
    logger.info(accelerator.state, main_process_only=False)

    config.world_size = accelerator.num_processes
    config.global_rank = accelerator.process_index
    if config.batch_size % accelerator.num_processes != 0:
        config.batch_size -= config.batch_size % accelerator.num_processes != 0
        logger.info('turn batch size to', config.batch_size)

    # Set random seed.
    accelerate.utils.set_seed(config.seed, device_specific=True)

    # load dataset
    dataset = datasets.load_dataset('train', config.data_dir, config)
    test_dataset = datasets.load_dataset('test', config.data_dir, config)
    dataloader = torch.utils.data.DataLoader(np.arange(len(dataset)),
                                             num_workers=8,
                                             shuffle=True,
                                             batch_size=1,
                                             collate_fn=dataset.collate_fn,
                                             persistent_workers=True,
                                             )
    test_dataloader = torch.utils.data.DataLoader(np.arange(len(test_dataset)),
                                                  num_workers=4,
                                                  shuffle=False,
                                                  batch_size=1,
                                                  persistent_workers=True,
                                                  collate_fn=test_dataset.collate_fn,
                                                  )
    if config.rawnerf_mode:
        postprocess_fn = test_dataset.metadata['postprocess_fn']
    else:
        postprocess_fn = lambda z, _=None: z

    dataset_info_for_model = {
        'size': dataset.size,
    }

    # setup model and optimizer
    model = models.Model(config=config, dataset_info=dataset_info_for_model)
    optimizer, lr_fn = train_utils.create_ft_optimizer(config, model)

    # reload model
    model = accelerator.prepare(model)
    checkpoints.restore_checkpoint(config.checkpoint_dir, accelerator, logger, strict=False)
    model = accelerator.unwrap_model(model)
    accelerator.clear() # reset accelerator

    # use accelerate to prepare.
    model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)

    module = accelerator.unwrap_model(model)
    dataiter = iter(dataloader)
    test_dataiter = iter(test_dataloader)

    num_params = train_utils.tree_len(list(model.parameters()))
    logger.info(f'Number of parameters being optimized: {num_params}')

    if (dataset.size > module.num_glo_embeddings and module.num_glo_features > 0):
        raise ValueError(f'Number of glo embeddings {module.num_glo_embeddings} '
                         f'must be at least equal to number of train images '
                         f'{dataset.size}')

    # metric handler
    metric_harness = image.MetricHarness()

    # tensorboard
    if accelerator.is_main_process:
        # function to convert image for tensorboard
        tb_process_fn = lambda x: x.transpose(2, 0, 1) if len(x.shape) == 3 else x[None]
    
    logger.info("Begin training...")
    init_step = 0
    total_time = 0
    total_steps = 0
    reset_stats = True
    num_steps = config.max_ft_steps
    
    with logging_redirect_tqdm():
        tbar = tqdm(range(init_step + 1, num_steps + 1),
                    desc='Training', initial=init_step, total=num_steps,
                    disable=not accelerator.is_main_process)

        for step in tbar:
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(dataloader)
                batch = next(dataiter)
            batch = accelerate.utils.send_to_device(batch, accelerator.device)
            if reset_stats and accelerator.is_main_process:
                stats_buffer = []
                train_start_time = time.time()
                reset_stats = False

            # use lr_fn to control learning rate
            learning_rate = lr_fn(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            # fraction of training period
            train_frac = np.clip((step - 1) / (config.max_ft_steps - 1), 0, 1)

            # Indicates whether we need to compute output normal or depth maps in 2D.
            compute_extras = (config.compute_disp_metrics or config.compute_normal_metrics)
            optimizer.zero_grad()
            with accelerator.autocast():
                renderings, ray_history = model(
                    False,  # Set rand to False to avoid nan.
                    batch,
                    train_frac=train_frac,
                    compute_extras=compute_extras,
                    zero_glo=False)

            losses = {}

            # supervised by data
            data_loss, stats = train_utils.compute_data_loss(batch, renderings, config)
            losses['data'] = data_loss

            # regularizations of bilateral grids
            if hasattr(module, 'bil_grids4d') and module.bil_grids4d is not None:
                if config.bilgrid4d_tv_loss_mult > 0:
                    losses['tv_bilgrids4d'] = train_utils.bilateral_grid4dcp_tv_loss(module, config)
               
            loss = sum(losses.values())
            stats['loss'] = loss.item()
            stats['losses'] = tree_map(lambda x: x.item(), losses)

            # accelerator automatically handle the scale
            accelerator.backward(loss)
            # clip gradient by max/norm/nan
            train_utils.clip_gradients(model, accelerator, config)
            optimizer.step()

            stats['psnrs'] = image.mse_to_psnr(stats['mses'])
            stats['psnr'] = stats['psnrs'][-1]

            # Log training summaries. This is put behind a host_id check because in
            # multi-host evaluation, all hosts need to run inference even though we
            # only use host 0 to record results.
            if accelerator.is_main_process:
                stats_buffer.append(stats)
                if step == init_step + 1 or step % config.print_every == 0:
                    elapsed_time = time.time() - train_start_time
                    steps_per_sec = config.print_every / elapsed_time
                    rays_per_sec = config.batch_size * steps_per_sec

                    # A robust approximation of total training time, in case of pre-emption.
                    total_time += int(round(TIME_PRECISION * elapsed_time))
                    total_steps += config.print_every
                    approx_total_time = int(round(step * total_time / total_steps))

                    # Transpose and stack stats_buffer along axis 0.
                    fs = [utils.flatten_dict(s, sep='/') for s in stats_buffer]
                    stats_stacked = {k: np.stack([f[k] for f in fs]) for k in fs[0].keys()}

                    # Split every statistic that isn't a vector into a set of statistics.
                    stats_split = {}
                    for k, v in stats_stacked.items():
                        if v.ndim not in [1, 2] and v.shape[0] != len(stats_buffer):
                            raise ValueError('statistics must be of size [n], or [n, k].')
                        if v.ndim == 1:
                            stats_split[k] = v
                        elif v.ndim == 2:
                            for i, vi in enumerate(tuple(v.T)):
                                stats_split[f'{k}/{i}'] = vi

                    # Take the mean and max of each statistic since the last summary.
                    avg_stats = {k: np.mean(v) for k, v in stats_split.items()}
                    max_stats = {k: np.max(v) for k, v in stats_split.items()}

                    precision = int(np.ceil(np.log10(config.max_ft_steps))) + 1
                    avg_loss = avg_stats['loss']
                    avg_psnr = avg_stats['psnr']
                    str_losses = {  # Grab each "losses_{x}" field and print it as "x[:4]".
                        k[7:11]: (f'{v:0.5f}' if 1e-4 <= v < 10 else f'{v:0.1e}')
                        for k, v in avg_stats.items()
                        if k.startswith('losses/')
                    }
                    logger.info(f'{step}' + f'/{config.max_ft_steps:d}:' +
                                f'loss={avg_loss:0.5f},' + f'psnr={avg_psnr:.3f},' +
                                f'lr={learning_rate:0.2e} | ' +
                                ','.join([f'{k}={s}' for k, s in str_losses.items()]) +
                                f',{rays_per_sec:0.0f} r/s')

                    # Reset everything we are tracking between summarizations.
                    reset_stats = True

                if step > 0 and step % config.checkpoint_every == 0 and accelerator.is_main_process:
                    checkpoints.save_checkpoint(config.ft_checkpoint_dir,
                                                accelerator, step,
                                                config.checkpoints_total_limit)

            # Test-set evaluation.
            if config.ft_render_every > 0 and step % config.ft_render_every == 0:
                # We reuse the same random number generator from the optimization step
                # here on purpose so that the visualization matches what happened in training.
                eval_start_time = time.time()
                try:
                    test_batch = next(test_dataiter)
                except StopIteration:
                    test_dataiter = iter(test_dataloader)
                    test_batch = next(test_dataiter)
                test_batch = accelerate.utils.send_to_device(test_batch, accelerator.device)

                # render a single image with all distributed processes
                rendering = models.render_image(model, accelerator,
                                                test_batch, False,
                                                train_frac, config,
                                                disable_bilgrid=True)

                # move to numpy
                rendering = tree_map(lambda x: x.detach().cpu().numpy(), rendering)
                test_batch = tree_map(lambda x: x.detach().cpu().numpy() if x is not None else None, test_batch)
                # Log eval summaries on host 0.
                if accelerator.is_main_process:
                    eval_time = time.time() - eval_start_time
                    num_rays = np.prod(test_batch['directions'].shape[:-1])
                    rays_per_sec = num_rays / eval_time

                    if config.vis_decimate > 1:
                        d = config.vis_decimate
                        decimate_fn = lambda x, d=d: None if x is None else x[::d, ::d]
                    else:
                        decimate_fn = lambda x: x
                    rendering = tree_map(decimate_fn, rendering)
                    test_batch = tree_map(decimate_fn, test_batch)
                    vis_start_time = time.time()
                    vis_suite = vis.visualize_suite(rendering, test_batch)
                    with tqdm.external_write_mode():
                        logger.info(f'Visualized in {(time.time() - vis_start_time):0.3f}s')
                    if config.rawnerf_mode:
                        # Unprocess raw output.
                        vis_suite['color_raw'] = rendering['rgb']
                        # Autoexposed colors.
                        vis_suite['color_auto'] = postprocess_fn(rendering['rgb'], None)
                        # Exposure sweep colors.
                        exposures = test_dataset.metadata['exposure_levels']
                        for p, x in list(exposures.items()):
                            vis_suite[f'color_exp{p}'] = postprocess_fn(rendering['rgb'], x)
                    if config.compute_normal_metrics:
                        utils.save_img_u8(test_batch['normals'] / 2. + 0.5, os.path.join(config.test_vis_dir, f'test_output_normals_{step:06d}.png'))
                    for k, v in vis_suite.items():
                        utils.save_img_u8(v, os.path.join(config.test_vis_dir, f'test_output_{k}_{step:06d}.png'))

    if accelerator.is_main_process and config.max_ft_steps > init_step:
        logger.info('Saving last checkpoint at step {} to {}'.format(step, config.ft_checkpoint_dir))
        checkpoints.save_checkpoint(config.ft_checkpoint_dir,
                                    accelerator, step,
                                    config.checkpoints_total_limit)
    logger.info('Finish training.')


if __name__ == '__main__':
    configs.define_common_flags()

    with gin.config_scope('train'):
        app.run(main)
