import glob
import logging
import os
import sys
import time

from absl import app
import gin
from internal import configs
from internal import datasets
from internal import models
from internal import train_utils
from internal import checkpoints
from internal import utils
from internal import vis
from matplotlib import cm
import mediapy as media
import torch
import numpy as np
import accelerate
import imageio
from torch.utils._pytree import tree_map



def create_videos(config, base_dir, out_dir, out_name, num_frames):
    """Creates videos out of the images saved to disk."""
    names = [n for n in config.exp_path.split(os.sep) if n]
    # Last two parts of checkpoint path are experiment name and scene name.
    exp_name, scene_name = names[-2:]
    video_prefix = f'{scene_name}_{exp_name}_{out_name}'

    zpad = max(3, len(str(num_frames - 1)))
    idx_to_str = lambda idx: str(idx).zfill(zpad)

    utils.makedirs(base_dir)

    # Load one example frame to get image shape and depth range.
    depth_file = os.path.join(out_dir, f'distance_mean_{idx_to_str(0)}.tiff')
    img_file = os.path.join(out_dir, f'color_{idx_to_str(0)}.png')
    if utils.file_exists(depth_file):
        depth_frame = utils.load_img(depth_file)
        shape = depth_frame.shape
        p = config.render_dist_percentile
        distance_limits = np.percentile(depth_frame.flatten(), [p, 100 - p])
        # lo, hi = [config.render_dist_curve_fn(x) for x in distance_limits]
        depth_curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
        lo, hi = distance_limits
    else:
        img_frame = utils.load_img(img_file)
        shape = img_frame.shape
    print(f'Video shape is {shape[:2]}')

    for k in ['color', 'normals', 'acc', 'distance_mean', 'distance_median']:
        video_file = os.path.join(base_dir, f'{video_prefix}_{k}.mp4')
        file_ext = 'png' if k in ['color', 'normals'] else 'tiff'
        file0 = os.path.join(out_dir, f'{k}_{idx_to_str(0)}.{file_ext}')
        if not utils.file_exists(file0):
            print(f'Images missing for tag {k}')
            continue
        print(f'Making video {video_file}...')

        writer = imageio.get_writer(video_file, fps=config.render_video_fps)
        for idx in range(num_frames):
            img_file = os.path.join(out_dir, f'{k}_{idx_to_str(idx)}.{file_ext}')
            if not utils.file_exists(img_file):
                ValueError(f'Image file {img_file} does not exist.')

            img = utils.load_img(img_file)
            if k in ['color', 'normals']:
                img = img / 255.
            elif k.startswith('distance'):
                # img = config.render_dist_curve_fn(img)
                # img = np.clip((img - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1)
                # img = cm.get_cmap('turbo')(img)[..., :3]

                img = vis.visualize_cmap(img, np.ones_like(img), cm.get_cmap('turbo'), lo, hi, curve_fn=depth_curve_fn)

            frame = (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)
            writer.append_data(frame)
        writer.close()


def main(unused_argv):
    config = configs.load_config()
    config.exp_path = os.path.join('exp', config.exp_name)
    config.checkpoint_dir = os.path.join(config.exp_path, 'checkpoints' if not config.render_ft else f'ft/{config.ft_name}/checkpoints/')
    config.render_dir = os.path.join(config.exp_path, 'render' if not config.render_ft else f'ft/{config.ft_name}/render')

    accelerator = accelerate.Accelerator()
    # setup logger
    logging.basicConfig(
        format="%(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(os.path.join(config.exp_path, 'log_render.txt' if not config.render_ft else f'ft/{config.ft_name}/log_render.txt'))],
        level=logging.INFO,
    )
    sys.excepthook = utils.handle_exception
    logger = accelerate.logging.get_logger(__name__)
    logger.info(config)
    logger.info(accelerator.state, main_process_only=False)

    config.world_size = accelerator.num_processes
    config.global_rank = accelerator.process_index
    accelerate.utils.set_seed(config.seed, device_specific=True)

    dataset = datasets.load_dataset('test', config.data_dir, config)
    dataloader = torch.utils.data.DataLoader(np.arange(len(dataset)),
                                             shuffle=False,
                                             batch_size=1,
                                             collate_fn=dataset.collate_fn,
                                             )
    dataiter = iter(dataloader)
    if config.rawnerf_mode:
        postprocess_fn = dataset.metadata['postprocess_fn']
    else:
        postprocess_fn = lambda z: z

    dataset_info_for_model = {
        'size': dataset.size,
    }
    model = models.Model(config=config, dataset_info=dataset_info_for_model)
    model.eval()

    model = accelerator.prepare(model)
    step = checkpoints.restore_checkpoint(config.checkpoint_dir, accelerator, logger, strict=config.load_state_dict_strict)

    logger.info(f'Rendering checkpoint at step {step}.')

    if config.render_path:
        out_name = 'path_renders'
    elif config.render_train:
        out_name = 'train_preds'
    else:
        out_name = 'test_preds'
    out_name = f'{out_name}_step_{step}'
    out_dir = os.path.join(config.render_dir, out_name)
    utils.makedirs(out_dir)

    path_fn = lambda x: os.path.join(out_dir, x)

    # Ensure sufficient zero-padding of image indices in output filenames.
    zpad = max(3, len(str(dataset.size - 1)))
    idx_to_str = lambda idx: str(idx).zfill(zpad)

    for idx in range(dataset.size):
        batch = next(dataiter)
        # If current image and next image both already exist, skip ahead.
        idx_str = idx_to_str(idx)
        curr_file = path_fn(f'color_{idx_str}.png')
        if utils.file_exists(curr_file):
            logger.info(f'Image {idx + 1}/{dataset.size} already exists, skipping')
            continue
        batch = tree_map(lambda x: x.to(accelerator.device) if x is not None else None, batch)
        logger.info(f'Evaluating image {idx + 1}/{dataset.size}')
        eval_start_time = time.time()
        rendering = models.render_image(model, accelerator,
                                        batch, False, 1, config)

        logger.info(f'Rendered in {(time.time() - eval_start_time):0.3f}s')

        if accelerator.is_main_process:  # Only record via host 0.
            rendering = tree_map(lambda x: x.detach().cpu().numpy() if x is not None else None, rendering)
            rendering['rgb'] = postprocess_fn(rendering['rgb'])
            utils.save_img_u8(rendering['rgb'], path_fn(f'color_{idx_str}.png'))
            if 'normals' in rendering:
                utils.save_img_u8(rendering['normals'] / 2. + 0.5,
                                  path_fn(f'normals_{idx_str}.png'))
            if not config.render_ft:
                utils.save_img_f32(rendering['distance_mean'],
                                   path_fn(f'distance_mean_{idx_str}.tiff'))
                utils.save_img_f32(rendering['distance_median'],
                                   path_fn(f'distance_median_{idx_str}.tiff'))
                if config.save_acc_maps:
                    utils.save_img_f32(rendering['acc'], path_fn(f'acc_{idx_str}.tiff'))

    num_files = len(glob.glob(path_fn('color_*.png')))
    if accelerator.is_main_process and num_files == dataset.size:
        logger.info(f'All files found, creating videos.')
        create_videos(config, config.render_dir, out_dir, out_name, dataset.size)
    accelerator.wait_for_everyone()
    logger.info('Finish rendering.')

if __name__ == '__main__':
    configs.define_common_flags()

    with gin.config_scope('eval'):  # Use the same scope as eval.py
        app.run(main)
