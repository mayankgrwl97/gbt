from math import ceil

import torch
import torch.nn as nn

from gbt.model.resnet import ResNetConv
from gbt.model.transformer import (TransformerDecoderLayerNoSelfAttn,
                                   TransformerDecoderMemoryMaskLayerwise,
                                   TransformerEncoderMaskLayerwise)
from gbt.model.utils import plucker_dist, transform_rays
from gbt.renderer.rays import (get_grid_rays, get_patch_rays,
                               get_plucker_parameterization,
                               get_random_query_pixel_rays,
                               positional_encoding)


class GeometryModule(nn.Module):
    """Computes a geometric distance between rays and yields an additive attention mask.

    Note: according to nn.MultiHeadAttention we have the following:
    (see https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward)

        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
        (L,S) or (N⋅num_heads,L,S), where N is the batch size, L is the
        target sequence length, and SS is the source sequence length. A 2D mask will be broadcasted across the
        batch while a 3D mask allows for a different mask for each entry in the batch. Binary, byte, and float
        masks are supported. For a binary mask, a True value indicates that the corresponding position is not
        allowed to attend. For a byte mask, a non-zero value indicates that the corresponding position is not
        allowed to attend. For a float mask, the mask values will be added to the attention weight.

    The mask generated will be of shape (B, Q, P) where B is the batch size, Q is the number of query rays per
    element in the batch, and P is the number of key rays per element in the batch. To use this mask in an
    attention layer, make sure to reshape it to (B*H, Q, P) where H is the number of heads in MHA. Use repeat
    interleave as in this example:

    https://github.com/pytorch/pytorch/blob/c7c2578f93fbfad5f769543848642a16b6071756/test/test_nn.py#L5710-L5716

    """

    def __init__(self, cfg):
        super(GeometryModule, self).__init__()

        if 'learnable_geometry' in cfg:
            self.learnable_geometry = cfg.learnable_geometry
        else:
            self.learnable_geometry = True  # Hard coded to be true

        self.geometric_weight = \
            nn.Parameter(data=torch.FloatTensor(cfg.geometric_weight_init_value), requires_grad=self.learnable_geometry)
        self.eps = cfg.geometric_distance_eps

    def forward(self, ray_q, ray_k, num_heads=None):
        """Yields an attention mask given queries and keys

        Args:
            ray_q (torch.Tensor): Tensor of shape (B, Q, 6) containing plucker representation of Q query rays
            ray_k (torch.Tensor): Tensor of shape (B, K, 6) containing plucker representation of K key rays
            num_heads (int|None): If provided, broadcasts the output to shape (B*H, Q, K) where H denotes the number of
                                  heads in the attention module.

        Returns:
            torch.Tensor: Tensor of shape (B, Q, K) or (B*H, Q, K) having the attention mask based on geometric
            distance. If num_heads is provided, the output is of shpae (B*H, Q, K) with repeat_interleave, otherwise
            the output is of shape (B, Q, K).
        """
        dists = plucker_dist(ray_q, ray_k, eps=self.eps)  # (B, Q, K)

        if num_heads is not None:
            # repeat_interleave is used as in this example:
            # https://github.com/pytorch/pytorch/blob/c7c2578f93fbfad5f769543848642a16b6071756/test/test_nn.py#L5710-L5716
            dists = torch.repeat_interleave(dists, num_heads, dim=0)  # (B*H, Q, K)

        dists_inv = [-(weight ** 2) * dists for weight in self.geometric_weight]
        dists_inv = torch.stack(dists_inv, dim=0) # (num_layers, B*H, Q, K)
        if len(dists_inv) == 1: # Same weight for all layers
            dists_inv = dists_inv[0]

        return dists_inv


class FeatureExtractor(nn.Module):
    """
    Takes in a set of input views from a scene and computes patch-wise image features using pretrained networks
    """
    def __init__(self, cfg):
        super(FeatureExtractor, self).__init__()

        # Following variables depend on the choice of network, and input image size
        self.image_feature_dim = cfg.image_feature_dim # Changing depth of network (n_blocks) will affect this
        self.num_patches_x = cfg.num_patches_x # Changing depth of network (n_blocks) and input image size will affect this
        self.num_patches_y = cfg.num_patches_y # Changing depth of network (n_blocks) and input image size will affect this
        self.n_blocks = cfg.n_blocks

        self.num_patches = self.num_patches_x * self.num_patches_y

        if 'use_feature_pyramid' in cfg:
            self.use_feature_pyramid = cfg.use_feature_pyramid
        else:
            self.use_feature_pyramid = False

        self.model = ResNetConv(n_blocks=self.n_blocks, use_feature_pyramid=self.use_feature_pyramid,
                                num_patches_x=self.num_patches_x, num_patches_y=self.num_patches_y)

    def forward(self, input_views):
        """
        Args:
            input_views: Input sparse views to the image feature extractor. (B, num_input_views, C, H, W)
        Returns:
            torch.Tensor: Tensor of shape (B, num_input_views, num_patches, image_feature_dim)
                          containing per-patch image features

        """
        b, n_inp, c, h, w = input_views.shape

        # Extract patch image features from input views
        image_features = self.model(torch.reshape(input_views, (b * n_inp, c, h, w)))  # (b * n_inp, feature_dim, patch_x, patch_y)
        image_features = image_features.reshape(b * n_inp, self.image_feature_dim, self.num_patches)  # (b*n_inp, feature_dim, patch)
        image_features = image_features.permute(0, 2, 1)  # (b*n_inp, patch, feature_dim)
        image_features = torch.reshape(image_features, (b, n_inp, self.num_patches, -1))  # (b, n_inp, patch, feature_dim)

        assert image_features.shape[-1] == self.image_feature_dim
        assert image_features.shape[-2] == self.num_patches

        return image_features


class SceneEncoder(nn.Module):
    """
    Takes set of patch-wise image and ray features as input and computes a set latent encoding for the scene
    """
    def __init__(self, cfg):
        super(SceneEncoder, self).__init__()

        # Transformer architecture params
        self.transformer_dim = cfg.transformer_dim
        self.encoder_hidden_activation = 'gelu'
        self.encoder_n_attention_heads = 12
        self.encoder_num_layers = cfg.num_encoder_layers

        self.use_geometry = cfg.use_geometry
        if self.use_geometry:
            self.geometry_module = GeometryModule(cfg)

        self.transformer_encoder = TransformerEncoderMaskLayerwise(
            nn.TransformerEncoderLayer(d_model=self.transformer_dim, nhead=self.encoder_n_attention_heads,
                                       activation=self.encoder_hidden_activation),
            num_layers=self.encoder_num_layers)
        return

    def log_scalars(self):
        scalars = {}
        if self.use_geometry:
            for i, weight in enumerate(self.geometry_module.geometric_weight):
                scalars[f'model/scene_encoder/geometric_weight_{i}'] = weight.item()
        return scalars

    def forward(self, scene_features_tuple):
        """
        Args:
            scene_features_tuple(tuple): Patch-wise concatenated image and ray features. [(b, n_inp, patch, transformer_dim), (b, n_inp, patch, 6)]
                                         (scene_features, input_patch_rays_plucker)
            src_mask(torch.Tensor): FloatTensor (additive mask) of shape (b * n_heads, n_inp * patch, n_inp * patch)
        Returns:
            torch.Tensor: Tensor of shape (n_inp*patch, b, d_model) representing scene latent encoding
        """
        scene_features, input_patch_rays_plucker = scene_features_tuple # (b, n_inp, patch, transformer_dim), (b, n_inp, patch, 6)
        input_patch_rays_plucker = input_patch_rays_plucker.flatten(start_dim=1, end_dim=2) # (b, num_inp*patch, 6)

        if self.use_geometry:
            src_mask = self.geometry_module(input_patch_rays_plucker,
                input_patch_rays_plucker, num_heads=self.encoder_n_attention_heads)  # (B*H, V*P, V*P)
        else:
            src_mask = None

        b, n_inp, n_patch, _ = scene_features.shape
        encoder_input = torch.reshape(scene_features, (b, n_inp * n_patch, self.transformer_dim))  # (b, n_inp*patch, d_model)
        encoder_input = encoder_input.permute(1, 0, 2)  # (n_inp*patch, b, d_model)
        scene_encoding = self.transformer_encoder(encoder_input, mask=src_mask)  # (n_inp*patch, b, d_model)
        return scene_encoding


class RayDecoder(nn.Module):
    """
    Decodes color value for each query pixel ray using a set latent encoding
    """
    def __init__(self, cfg):
        super(RayDecoder, self).__init__()

        # Transformer architecture params
        self.transformer_dim = cfg.transformer_dim
        self.decoder_hidden_activation = 'gelu'
        self.decoder_n_attention_heads = 12
        self.decoder_num_layers = cfg.num_decoder_layers

        self.use_geometry = cfg.use_geometry
        if self.use_geometry:
            self.geometry_module = GeometryModule(cfg)

        self.transformer_decoder = TransformerDecoderMemoryMaskLayerwise(
            TransformerDecoderLayerNoSelfAttn(d_model=self.transformer_dim, nhead=self.decoder_n_attention_heads,
                                              activation=self.decoder_hidden_activation),
            num_layers=self.decoder_num_layers)

        self.rgb_mlp = nn.Sequential(
            nn.Linear(self.transformer_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def log_scalars(self):
        scalars = {}
        if self.use_geometry:
            for i, weight in enumerate(self.geometry_module.geometric_weight):
                scalars[f'model/ray_decoder/geometric_weight_{i}'] = weight.item()
        return scalars

    def forward(self, query_pixel_rays_tuple, scene_encoding_tuple):
        """
        Args:
            query_pixel_rays_tuple (tuple): Rays corresponding to a pixel in the image [(b, num_pixel_queries, transformer_dim), (b, num_pixel_queries, 6)]
                                            (query_pixel_rays, query_pixel_rays_plucker)
            scene_encoding_tuple (tuple): Set latent representation [(n_inp * patch, b, transformer_dim), (b, num_inp, patch, 6)]
                                          (scene_encoding, input_patch_rays_plucker)
            memory_mask: Tensor of shape (b * n_heads, num_queries, n_inp * patch)
        Returns:
            torch.Tensor: Tensor of shape (b, num_pixel_queries, 3) representing rgb value for each input ray
        """
        query_pixel_rays, query_pixel_rays_plucker = query_pixel_rays_tuple # (b, num_pixel_queries, transformer_dim), (b, num_pixel_queries, 6)
        scene_encoding, input_patch_rays_plucker = scene_encoding_tuple # (n_inp * patch, b, transformer_dim), (b, num_inp, patch, 6)

        input_patch_rays_plucker = input_patch_rays_plucker.flatten(start_dim=1, end_dim=2) # (b, num_inp*patch, 6)

        if self.use_geometry:
            memory_mask = self.geometry_module(query_pixel_rays_plucker,
                input_patch_rays_plucker, num_heads=self.decoder_n_attention_heads)  # (B*H, Q, V*P)
        else:
            memory_mask = None

        # Decode query rays using scene latent representation
        query_pixel_rays = query_pixel_rays.permute(1, 0, 2)  # (num_pixel_queries, b, d)
        pred_embed = self.transformer_decoder(query_pixel_rays, scene_encoding, tgt_mask=None,
                                              memory_mask=memory_mask)  # (num_pixel_queries, b, d_model)
        pred_embed = pred_embed.permute(1, 0, 2)  # (b, num_pixel_queries, d_model)

        # Predict pixel rgb color values
        pred_pixels = self.rgb_mlp(pred_embed)  # (b, num_pixel_queries, 3)
        return pred_pixels


class GeometryBiasedTransformer(nn.Module):
    """
    A "scene" represents a novel scene/object in 3D, and our input consists of multiple sparse views
    (num_input_views) corresponding to that scene. Aim is to form a scene embedding for input sparse
    view images and patch rays corresponding to these images. We pass the input images to a pre-trained
    feature extractor (could be anything, ViT or ResNet) to obtain patch embeddings of shape
    (num_input_views*P, D1). We also encode corresponding patch rays (corresponding to center pixel
    of each patch) of shape (num_input_views*P, D2). We concatenate the image and ray embeddings
    (num_input_views*P, D1+D2) and pass them to a transformer encoder to generate a scene encoding of
    dimensions - (num_input_views*P, D).

    The scene encoding from the transformer encoder is fed to another transformer decoder along with
    per-pixel query rays from a novel view point to generate novel view pixel values. We will then
    take a reconstruction loss between the predicted pixels and gt pixels.

    """

    def __init__(self, cfg):
        super(GeometryBiasedTransformer, self).__init__()

        self.num_pixel_queries = cfg.num_pixel_queries

        # Image patch feature extractor
        self.feature_extractor = FeatureExtractor(cfg.feature_extractor)
        self.image_feature_dim = self.feature_extractor.image_feature_dim
        self.num_patches_x = self.feature_extractor.num_patches_x
        self.num_patches_y = self.feature_extractor.num_patches_y

        # Ray positional encoding args
        self.num_freqs = cfg.ray.num_freqs
        self.start_freq = cfg.ray.start_freq
        self.parameterize = cfg.ray.parameterize
        self.harmonic_embedding_dim = 2 * self.num_freqs * 6
        self.view_space = cfg.ray.view_space

        # Transformer encoder and decoder
        self.transformer_dim = cfg.transformer_dim

        self.scene_encoder = SceneEncoder(cfg.scene_encoder)
        # To map (image_feature | patch_ray_embedding) -> (transformer_dim)
        self.linear_scene = nn.Linear(
            self.image_feature_dim + self.harmonic_embedding_dim, self.transformer_dim)

        self.ray_decoder = RayDecoder(cfg.ray_decoder)

        # To map (query_ray_embedding) -> (transformer_dim)
        self.linear_query_pixel_rays = nn.Linear(self.harmonic_embedding_dim, self.transformer_dim)

    def get_query_rays(self, query_cameras, image_size=None, query_ray_filter=None):
        if self.training:
            return get_random_query_pixel_rays(
                query_cameras, num_pixel_queries=self.num_pixel_queries, query_ray_filter=query_ray_filter,
                min_x=1, max_x=-1, min_y=1, max_y=-1,
                return_xys=True, device='cpu')  # Setting default to cpu. Device will be handled by each Dataparallel process.
        else:
            return get_grid_rays(
                query_cameras, image_size=image_size,
                min_x=1, max_x=-1, min_y=1, max_y=-1,
                device='cpu')  # Setting default to cpu. Device will be handled by each Dataparallel process.

    def infer(self, input_views, input_cameras, query_cameras, image_size=None):
        """Infers model for a given set of input views and the query view. Predicts the pixel values for all pixels (H*W) given the query view.
        Args:
            input_views(torch.Tensor): Tensor of shape (B, num_input_views, C, H, W) containing the images corresponding to the input views.
            input_cameras(list[pytorch3d.renderer.cameras.CamerasBase]): List of Pytorch3D cameras of length (n_cameras,) corresponding to the
                                                                        input views.
            query_cameras(list[pytorch3d.renderer.cameras.CamerasBase]): List of Pytorch3D cameras of length (n_query_cameras,) corresponding to the
                                                                        query views.
            image_size(tuple[int, int]): Size of the image in pixels (height, width).

        Returns:
            torch.Tensor: Tensor of shape (n_query_cameras, H*W, 3) containing the predicted
                        pixel values for each pixel in each query view.
        """
        assert not self.training, "Set model.eval() before calling infer"

        with torch.no_grad():
            grid_rays, _ = self.get_query_rays(
                query_cameras, image_size=image_size)  # grid_rays: (n_query_cameras, H*W, 6), and, xys: (n_query_cameras, H*W, 2)

            # Break the given number of query rays into reasonable batches (to avoid OOM)
            n_queries = self.num_pixel_queries
            num_query_batches = ceil(grid_rays.shape[1] / n_queries)
            pred_pixels_list = []
            for i in range(num_query_batches):
                # Get grid rays corresponding to the current batch of pixels
                grid_rays_current_iter = grid_rays[:, i * n_queries:(i + 1) * n_queries]  # (n_query_cameras, n_queries, 6)

                # Predict the pixel values for the given rays
                pred_pixels = self(input_indices=torch.arange(input_views.shape[0]),
                                   input_views=input_views,
                                   input_cameras=input_cameras,
                                   query_pixel_rays=grid_rays_current_iter)  # (n_query_cameras, n_queries, 3)
                pred_pixels_list.append(pred_pixels)

            # Concatenate all query outputs for each item in the batch to get the predicted pixel colors
            pred_pixels = torch.cat(pred_pixels_list, dim=1)  # (n_query_cameras, H*W, 3)

        # Notes:
        # The shape of pred_pixels: (n_query_cameras, H*W, 3) where H*W is the number of rays yielded by get_grid_rays above
        return pred_pixels

    def convert_to_view_space(self, input_cameras, input_rays, query_rays):
        if not self.view_space:
            return input_rays, query_rays

        reference_cameras = [cameras[0] for cameras in input_cameras]
        reference_R = [camera.R.to(input_rays.device) for camera in reference_cameras] # List (length=batch_size) of Rs(shape: 1, 3, 3)
        reference_R = torch.cat(reference_R, dim=0) # (B, 3, 3)
        reference_T = [camera.T.to(input_rays.device) for camera in reference_cameras] # List (length=batch_size) of Ts(shape: 1, 3)
        reference_T = torch.cat(reference_T, dim=0) # (B, 3)
        input_rays = transform_rays(reference_R=reference_R, reference_T=reference_T, rays=input_rays)
        query_rays = transform_rays(reference_R=reference_R, reference_T=reference_T, rays=query_rays.unsqueeze(1)).squeeze(1)
        return input_rays, query_rays

    def log_scalars(self):
        scalars = {}
        scalars.update(self.ray_decoder.log_scalars())
        scalars.update(self.scene_encoder.log_scalars())
        return scalars

    def forward(self, input_indices, input_views, input_cameras=None, query_pixel_rays=None):
        """
        Args:
            input_indices: A Tensor of indices [0, 1, 2, 3 .. B-1] (used for splitting batch according to dataparallel)
            input_views: Input sparse views to the image feature extractor. (B, num_input_views, C, H, W)
            input_cameras: Input cameras corresponding to each provided view for the batch (list of list cameras; list shape (B, num_input_views)).
            query_pixel_rays: Rays corresponding to the pixels that we want to infer rgb values for. (B, num_pixel_queries, 6) - note: (origin, direction) representation

        Returns:
            torch.Tensor: Predicted pixel values corresponding to query_pixel_rays of shape (B, num_pixel_queries, 3).

        """
        # Split input cameras according to the indices
        device = input_views.device
        input_cameras = [input_cameras[i] for i in input_indices]
        if query_pixel_rays.device != device:
            query_pixel_rays = query_pixel_rays.to(device)

        # Compute patch-wise image features for all input views
        image_features = self.feature_extractor(input_views)  # (b, n_inp, patch, img_feature_dim)

        # Compute rays corresponding to center of patch for each input view,
        # normalized (origin, direction) ray representation
        input_patch_rays = get_patch_rays(input_cameras, num_patches_x=self.num_patches_x,
                                          num_patches_y=self.num_patches_y, device=input_views.device) # (b, n_inp, patch, 6)

        # Convert to view space if specified during model instantiation
        input_patch_rays, query_pixel_rays = self.convert_to_view_space(
            input_cameras, input_patch_rays, query_pixel_rays)

        # Keep a copy of input patch rays and query pixel rays
        # Convert to plucker ray representation
        input_patch_rays_plucker = get_plucker_parameterization(input_patch_rays)  # (b, n_inp, patch, 6)
        query_pixel_rays_plucker = get_plucker_parameterization(query_pixel_rays)  # (b, num_pixel_queries, 6)

        # Convert to plucker and convert to harmonics embeddings
        input_patch_rays = positional_encoding(input_patch_rays, n_freqs=self.num_freqs,
            parameterize=self.parameterize, start_freq=self.start_freq)  # (b, n_inp, patch, pos_embedding_dim)
        # Concatenate input patch ray embeddings to image patch features
        scene_features = torch.cat((image_features, input_patch_rays), dim=-1)  # (b, n_inp, patch, img_feature_dim + pos_embedding_dim)

        # Project scene features to transformer dimensions
        scene_features = self.linear_scene(scene_features)  # (b, n_inp, patch, transformer_dim)

        # Scene latent representation
        scene_encoding = self.scene_encoder((scene_features, input_patch_rays_plucker))  # (n_inp * patch, b, transformer_dim)

        # Encode and project query rays to transformer dimensions
        query_pixel_rays = positional_encoding(query_pixel_rays, n_freqs=self.num_freqs,
            parameterize=self.parameterize, start_freq=self.start_freq)  # (b, num_pixel_queries, pos_embedding_dim)
        query_pixel_rays = self.linear_query_pixel_rays(query_pixel_rays)  # (b, num_pixel_queries, transformer_dim)

        pred_pixels = self.ray_decoder((query_pixel_rays, query_pixel_rays_plucker), (scene_encoding, input_patch_rays_plucker))
        return pred_pixels
