import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import lpips
import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from torch.utils.data import DataLoader
from tqdm import tqdm

from gbt.dataset.co3d_dataset import Co3dV2Dataset
from gbt.model.model import GeometryBiasedTransformer
from gbt.utils.image import (convert_tensor_to_images, save_images_as_gif,
                             stitch_images)


def infer(model, dataloader, infer_cfg, visdir):
    # Load Models
    device = infer_cfg.device
    lpips_vgg = lpips.LPIPS(net='vgg').to(device).eval()
    model.load_state_dict(torch.load(infer_cfg.load_path))
    model.to(device).eval()

    how_many = infer_cfg.get('how_many', int(1e6))
    image_size = tuple(dataloader.dataset.image_size)

    # Set up paths
    metrics_path = os.path.join(visdir, 'metrics.txt')
    with open(metrics_path, 'a') as f:
        f.writelines(f'obj_id;psnr;lpips\n')

    all_objs_psnr = []
    all_objs_lpips = []
    for batch_id, batch in tqdm(enumerate(dataloader)):
        obj_psnr = []
        obj_lpips = []

        input_views = batch["sparse_input_images"].to(device)  # (B, num_input_views, C, H, W)
        input_cameras = batch["sparse_input_cameras"]  # 2-d list of cameras of shape (B, num_input_views)
        query_img = batch["sparse_query_images"].to(device)  # (B, num_query_views, C, H, W)
        query_cameras = batch["sparse_query_cameras"]  # 2-d list of cameras of shape (B, num_query_views)
        obj_ids = batch['obj_id']  # List of object ids corresponding to objects in the batch

        num_query_views = query_img.shape[1]

        stitched_images = []
        for q_idx in range(num_query_views):
            pred_img = model.infer(input_views=input_views[:1], input_cameras=input_cameras[:1],
                                   query_cameras=[query_cameras[0][q_idx]], image_size=image_size)  # (1, H*W, 3)
            pred_img = pred_img.reshape(1, *image_size, 3) # (1, H, W, 3)

            # (N, H, W, 3) - [0, 1]
            input_image, query_image, pred_image = convert_tensor_to_images(
                input_views[0], query_img[:1, q_idx], pred_img)

            psnr = calc_psnr(query_image, pred_image, data_range=1.0)
            obj_psnr.append(psnr)
            lpips_metric = lpips_vgg(
                pred_img.permute(0, 3, 1, 2) * 2 - 1, # pred image: normalized to [-1,1]
                query_img[0][q_idx:q_idx+1] * 2 - 1 # gt image: normalized to [-1,1]
            ).detach().squeeze().item()
            obj_lpips.append(lpips_metric)

            stitched_image = stitch_images(input_image, query_image, pred_image)
            stitched_image = (stitched_image * 255).astype(np.uint8)
            stitched_images.append(stitched_image)

        avg_obj_psnr = np.mean(obj_psnr)
        avg_obj_lpips = np.mean(obj_lpips)
        all_objs_psnr.append(avg_obj_psnr)
        all_objs_lpips.append(avg_obj_lpips)
        with open(metrics_path, 'a') as f:
            f.writelines(f'{obj_ids[0]};{avg_obj_psnr:.2f};{avg_obj_lpips:.2f}\n')
        print(np.mean(all_objs_psnr), np.mean(all_objs_lpips))

        # Save to disk
        if batch_id % infer_cfg.visualize_after == 0:
            out_path = os.path.join(visdir, f'{obj_ids[0]}.gif')
            save_images_as_gif(save_path=out_path, images=stitched_images, fps=15)
            print(f'Visualilzation saved at -- {out_path}')

        if batch_id + 1 == how_many:
            break

    avg_psnr = np.mean(all_objs_psnr)
    avg_lpips = np.mean(all_objs_lpips)
    with open(metrics_path, 'a') as f:
        f.writelines(f'Average;{avg_psnr:.2f};{avg_lpips:.2f}\n')


def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--dataset-path', type=str, default=None)
    parser.add_argument('--category', type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    if args.dataset_path is not None:
        cfg.val_dataset.path = args.dataset_path
    if args.category is not None:
        cfg.val_dataset.category = args.category

    print(OmegaConf.to_yaml(cfg))
    return cfg


if __name__ == '__main__':
    cfg = get_cfg()

    # Load pre-trained model
    model = GeometryBiasedTransformer(cfg.model)

    if cfg.val_dataset.type == 'co3dv2':
        val_dataset = Co3dV2Dataset(cfg.val_dataset)
        collate_fn = Co3dV2Dataset.collate_fn
    else:
        raise NotImplementedError("Currently only supported - co3dv2 dataset")

    dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=cfg.infer.num_workers,
                            pin_memory=True, collate_fn=collate_fn)

    visdir = os.path.join(cfg.training.runs_dir, cfg.training.exp_tag, 'infer',
                          f"num_views={cfg.val_dataset.num_input_views}", cfg.val_dataset.category)
    os.makedirs(visdir, exist_ok=True)
    infer(model, dataloader, cfg.infer, visdir)
