import cv2
import imageio
import numpy as np


def convert_tensor_to_images(input_views, query_img, pred_img):
    """
    input_views: (V, 3, H, W), pixel range: [-1, 1]
    query_img: (1, 3, H, W), pixel range: [0, 1]
    pred_img: (1, H, W, 3), pixel range: [0, 1]
    """
    input_image = np.transpose(input_views.cpu().numpy(), (0, 2, 3, 1)) * 0.5 + 0.5 # (V, H, W, 3) - [0, 1]
    query_image = np.transpose(query_img.cpu().numpy(), (0, 2, 3, 1)) # (1, H, W, 3) - [0, 1]
    pred_image = pred_img.cpu().numpy() # (1, H, W, 3) - [0, 1]
    return input_image, query_image, pred_image

def stitch_images(input_image, query_image, pred_image):
    """
    input_image: (V, H, W, 3) - [0, 1]
    query_image: (1, H, W, 3) - [0, 1]
    pred_image: (1, H, W, 3) - [0, 1]
    """
    num_input_images = input_image.shape[0]
    _, hq, wq, _ = query_image.shape

    # Assuming images are numpy arrays of shape N x H x W x 3
    input_image = np.concatenate(list(input_image), axis=1) # H x Ni*W x 3
    if input_image.shape[1:3] != query_image.shape[1:3]:
        input_image = cv2.resize(input_image, dsize=(wq * num_input_images, hq))
    query_image = np.concatenate(list(query_image), axis=1) # H x Nq*W x3
    pred_image = np.concatenate(list(pred_image), axis=1) # H x Np*W x 3

    stitched_image = np.concatenate([input_image, query_image, pred_image], axis=1)
    return stitched_image

def save_images_as_gif(save_path, images, fps=5):
    imageio.mimsave(save_path, images, fps=fps)