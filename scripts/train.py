import argparse
import copy
import os
import shutil
import sys

import numpy as np
import submitit
import torch
from omegaconf import OmegaConf
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gbt.dataset.co3d_dataset import Co3dV2Dataset
from gbt.model.model import GeometryBiasedTransformer
from gbt.utils.model import ModuleDataParallel, load_model
from gbt.utils.image import (convert_tensor_to_images, sample_images_at_xy,
                             stitch_images)


def validate(model, val_dataloader, device, visdir, steps, num_scenes=10):
    model.eval()
    image_size = tuple(val_dataloader.dataset.image_size)

    visdir = os.path.join(visdir, f'steps_{steps}')
    os.makedirs(visdir, exist_ok=True)

    for batch_id, batch in enumerate(val_dataloader):
        input_views = batch["sparse_input_images"].to(device)  # (B, num_input_views, C, H, W)
        input_cameras = batch["sparse_input_cameras"]  # 2-d list of cameras of shape (B, num_input_views)
        query_img = batch["sparse_query_images"].to(device)  # (B, num_query_views, C, H, W)
        query_cameras = batch["sparse_query_cameras"]  # 2-d list of cameras of shape (B, num_query_views)

        num_query_views = query_img.shape[1]

        for q_idx in range(num_query_views):
            pred_img = model.infer(input_views=input_views[:1], input_cameras=input_cameras[:1],
                                   query_cameras=[query_cameras[0][q_idx]], image_size=image_size)  # (1, H*W, 3)

            input_image, query_image, pred_image = convert_tensor_to_images(
                input_views[0], query_img[:1, q_idx], pred_img.reshape(1, *image_size, 3))

            stitched_image = stitch_images(input_image, query_image, pred_image)

            # Save to disk
            out_path = os.path.join(visdir, f'GENERATED_OUTPUT_{batch_id}_{q_idx}.png')
            Image.fromarray((stitched_image*255).astype(np.uint8)).save(out_path)

        if batch_id >= num_scenes:
            break
    return


def train(model, train_dataloader, val_dataloader, cfg):
    # Create optimizer
    optim = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)

    # Set up mixed precision
    if cfg.use_mixed_precision:
        print('Training with mixed precision.')
    else:
        print('Training with full precision. Set use_mixed_precision=True for mixed precision training.')
    gradscaler = torch.cuda.amp.GradScaler(enabled=cfg.use_mixed_precision)

    # Set up directories
    exp_dir = os.path.join(cfg.runs_dir, cfg.exp_tag)
    logdir = os.path.join(exp_dir, 'logs')
    ckpt_dir = os.path.join(exp_dir, 'ckpts')
    train_visdir = os.path.join(exp_dir, 'train_vis')
    val_visdir = os.path.join(exp_dir, 'val_vis')
    for d in ([train_visdir, val_visdir, ckpt_dir]):
        os.makedirs(d, exist_ok=True)

    # Set up logger
    writer = SummaryWriter(logdir=logdir)
    writer.add_text('cfg', str(OmegaConf.to_yaml(cfg)))

    # Load training state (continue training)
    ckpt_path = os.path.join(ckpt_dir, 'latest.pt')
    if cfg.load_path is not None:
        batch_id = load_model(cfg.load_path, model.module, optim, gradscaler)
    elif os.path.exists(ckpt_path):  # fallback to loading latest checkpoint
        batch_id = load_model(ckpt_path, model.module, optim, gradscaler)
    else:
        batch_id = 1

    # ==================
    # Main Training loop
    # ==================
    device = cfg.device
    loss_fn = torch.nn.MSELoss()

    for epoch in tqdm(range(cfg.num_epochs)):
        for batch in train_dataloader:
            model.train()

            # Load data
            input_views = batch["sparse_input_images"].to(device)  # (B, num_input_views, C, H, W)
            input_cameras = [x for x in batch["sparse_input_cameras"]]  # 2-d list of cameras of shape (B, num_input_views)
            query_img = batch["sparse_query_images"].to(device)  # (B, C, H, W)
            query_cameras = batch["sparse_query_cameras"]  # list of query cameras (B,)
            query_ray_filter = batch.get("sparse_query_crop_masks", None)  # (B, H, W) - [0/1] binary mask

            query_pixel_rays, query_pixel_ndc = \
                model.get_query_rays(query_cameras, query_ray_filter=query_ray_filter)   # (B, num_pixel_queries, 6),  (B, num_pixel_queries, 2)
            query_pixel_ndc = query_pixel_ndc.to(query_img.device)  # Cast to query image device

            # Model forward pass
            with torch.cuda.amp.autocast(enabled=cfg.use_mixed_precision):
                pred_pixels = model(input_indices=torch.arange(input_views.shape[0]),
                                    input_views=input_views, input_cameras=input_cameras,
                                    query_pixel_rays=query_pixel_rays)  # (B, num_pixel_queries, 3)

                # Obtain the target pixels corresponding to the randomly predicted query pixels during forward pass
                query_pixels = sample_images_at_xy(query_img, query_pixel_ndc)  # (B, 3, num_pixel_queries)
                query_pixels = query_pixels.permute(0, 2, 1)  # (B, num_pixel_queries, 3) | Permutation required because model's output is (..., 3)

                loss = loss_fn(pred_pixels, query_pixels)

            optim.zero_grad()
            if cfg.use_mixed_precision:
                gradscaler.scale(loss).backward()
                gradscaler.step(optim)
                gradscaler.update()
            else:
                loss.backward()
                optim.step()

            writer.add_scalar('train/loss', loss, batch_id)
            writer.add_scalar('train/lr', optim.param_groups[0]['lr'], batch_id)
            scalars = model.log_scalars()
            for k, v in scalars.items():
                writer.add_scalar(k, v, batch_id)

            if batch_id % cfg.logging.visualize_after == 0:
                model.eval()

                image_size = tuple(train_dataloader.dataset.image_size)
                pred_img = model.infer(input_views=input_views[:1], input_cameras=input_cameras[:1],
                                       query_cameras=query_cameras[:1], image_size=image_size)  # (1, H*W, 3)

                input_image, query_image, pred_image = convert_tensor_to_images(
                    input_views[0], query_img[:1], pred_img.reshape(1, *image_size, 3))

                stitched_image = stitch_images(input_image, query_image, pred_image)  # [0-1, np.float32]

                # Save to disk
                out_path = os.path.join(train_visdir, f'GENERATED_OUTPUT_{batch_id}.png')
                Image.fromarray((stitched_image*255).astype(np.uint8)).save(out_path)

                # Log to tensorboard
                writer.add_image("input images (3), query image (1), pred image (1)", stitched_image, batch_id, dataformats='HWC')

            if batch_id % cfg.saving.save_after == 0:
                torch.save({
                    'batch_id': batch_id,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'scaler_state_dict': gradscaler.state_dict(),
                }, ckpt_path)

            batch_id += 1

        if epoch % cfg.validation.validate_after_epochs == 0:
            validate(model, val_dataloader, device, val_visdir, batch_id)

    return


def trainer(cfg):
    print(OmegaConf.to_yaml(cfg))
    model = GeometryBiasedTransformer(cfg.model).to(cfg.training.device)
    model = ModuleDataParallel(model, device_ids=cfg.training.device_ids)

    train_dataset = Co3dV2Dataset(cfg.train_dataset)
    val_dataset = Co3dV2Dataset(cfg.val_dataset)
    collate_fn = Co3dV2Dataset.collate_fn

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.training.batch_size,
                                  shuffle=True, num_workers=cfg.training.num_workers, pin_memory=True,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=cfg.training.num_workers,
                                pin_memory=True, collate_fn=collate_fn)

    train(model, train_dataloader, val_dataloader, cfg.training)


def get_cfg(cfg_path, verbose=True):
    cfg = OmegaConf.load(cfg_path)
    if verbose:
        print(OmegaConf.to_yaml(cfg))

    exp_dir = os.path.join(cfg.training.runs_dir, cfg.training.exp_tag)
    os.makedirs(exp_dir, exist_ok=True)
    to_path = os.path.join(exp_dir, os.path.basename(cfg_path))
    if not os.path.exists(to_path):
        shutil.copyfile(cfg_path, os.path.join(exp_dir, os.path.basename(cfg_path)))
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=True)
    args = parser.parse_args()
    cfg = get_cfg(args.config_path, verbose=True)
    trainer(cfg)
