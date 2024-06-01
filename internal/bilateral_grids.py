import gin
from internal import image
from internal import math
from internal import utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
tl.set_backend('pytorch')
from tensorly.decomposition import parafac


set_kwargs = utils.set_kwargs



def color_affine_transform(affine_mats, rgb):
    '''
    Apply per-pixel color affine transformations.
    :param affine_mats: affine transformation matrices (..., 3, 4).
    :param rgb: pixel RGB values (..., 3).
    :return: color transformed image (..., 3).
    '''
    return torch.matmul(affine_mats[..., :3], rgb.unsqueeze(-1)).squeeze(-1) + affine_mats[..., 3]


def slice(bil_grids, xy, rgb, grid_idx):
    '''
    Slice batch of bilateral grids. Suppose N is number of bilateral grids.
    Supports input shape: (chunk_size, dim), (num_patches, h, w, dim), (h, w, dim).

    :param bil_grids: the bilateral grids.
    :param xy: the x-y coordinates (..., 2).
    :param rgb: the corresponding RGB values (..., 3).
    :param grid_idx: grid indices for slicing (..., 1).
    '''
    sh_ = rgb.shape

    grid_idx_unique = torch.unique(grid_idx)
    if len(grid_idx_unique) == 1:
        # All pixels are from a single view.
        grid_idx = grid_idx_unique # (1,)
        xy = xy.unsqueeze(0) # (1, ..., 2)
        rgb = rgb.unsqueeze(0) # (1, ..., 3)
    else:
        # Pixels are randomly sampled from different views.
        if len(grid_idx.shape) == 4:
            grid_idx = grid_idx[:, 0, 0, 0] # (chunk_size,)
        elif len(grid_idx.shape) == 3:
            grid_idx = grid_idx[:, 0, 0] # (chunk_size,)
        elif len(grid_idx.shape) == 2:
            grid_idx = grid_idx[:, 0] # (chunk_size,)
        else:
            raise ValueError(f'The input to bilateral grid slicing is not supported yet.')

    affine_mats = bil_grids(xy, rgb, grid_idx)
    rgb = color_affine_transform(affine_mats, rgb)

    return {
        'rgb': rgb.reshape(*sh_),
        'rgb_affine_mats': affine_mats.reshape(*sh_[:-1], affine_mats.shape[-2], affine_mats.shape[-1])
    }


@gin.configurable
class BilateralGrid(nn.Module):
    grid_width = 16  # Grid width.
    grid_height = 16  # Grid height.
    grid_depth = 8  # Guidance dimension.

    def __init__(self, num, **kwargs):
        """
        :param num: number of bilateral grids (= # views).
        """
        super(BilateralGrid, self).__init__()
        set_kwargs(self, kwargs)
        
        # Initialize grids.
        grid = self.init_identity_grid()
        self.grids = nn.Parameter(grid.tile(num, 1, 1, 1, 1)) # (N, 12, D, H, W)

        # Weights of BT601 RGB-to-gray.
        self.register_buffer('rgb2gray_weight', torch.Tensor([[0.299, 0.587, 0.114]]))
        self.rgb2gray = lambda rgb: (rgb @ self.rgb2gray_weight.T) * 2. - 1.

    def init_identity_grid(self):
        grid = torch.tensor([1., 0, 0, 0, 0, 1., 0, 0, 0, 0, 1., 0,]).float()
        grid = grid.repeat([self.grid_depth * self.grid_height * self.grid_width, 1])  # (D * H * W, 12)
        grid = grid.reshape(1, self.grid_depth, self.grid_height, self.grid_width, -1) # (1, D, H, W, 12)
        grid = grid.permute(0, 4, 1, 2, 3)  # (1, 12, D, H, W)
        return grid

    def forward(self, grid_xy, rgb, idx=None):
        """
        Bilateral grid slicing.
        :param grid_xy: x-y coordinates (N, ..., 2).
                        When not using 5D input, the `idx` parameter should be specified.
        :param rgb: (N, ..., 3).
        :param idx: view indices (N,).
        :return: affine matrices (N, ..., 3, 4).
        """
        grids = self.grids
        input_ndims = len(grid_xy.shape)
        assert len(rgb.shape) == input_ndims

        if input_ndims > 1 and input_ndims < 5:
            # Convert input into 5D
            for i in range(5 - input_ndims):
                grid_xy = grid_xy.unsqueeze(1)
                rgb = rgb.unsqueeze(1)
            assert idx is not None
        elif input_ndims != 5:
            raise ValueError('Bilateral grid slicing only takes either 2D, 3D, 4D and 5D inputs')

        grids = self.grids
        if idx is not None:
            grids = grids[idx]
        assert grids.shape[0] == grid_xy.shape[0]
        
        # Generate slicing coordinates.
        grid_xy = (grid_xy - 0.5) * 2 # Rescale to [-1, 1].
        grid_z = self.rgb2gray(rgb)
        grid_xyz = torch.cat([grid_xy, grid_z], dim=-1) # (N, m, h, w, 3)

        affine_mats = F.grid_sample(grids, grid_xyz, mode='bilinear', align_corners=True, padding_mode='border')  # (N, 12, m, h, w)
        affine_mats = affine_mats.permute(0, 2, 3, 4, 1)  # (N, m, h, w, 12)
        affine_mats = affine_mats.reshape(*affine_mats.shape[:-1], 3, 4)  # (N, m, h, w, 3, 4)

        for _ in range(5 - input_ndims):
            affine_mats = affine_mats.squeeze(1)
        
        return affine_mats
    

def slice4d(bil_grids4d, xyz, rgb):
    '''
    :param bil_grids4d: the 4D bilateral grids.
    :param xyz: the xyz coordinates (..., 3).
    :param rgb: the corresponding RGB values (..., 3).
    '''
    affine_mats = bil_grids4d(xyz, rgb)
    rgb = color_affine_transform(affine_mats, rgb)

    return {
        'rgb': rgb,
        'rgb_affine_mats': affine_mats
    }


class _ScaledTanh(nn.Module):
    def __init__(self, s=2.0):
        super(_ScaledTanh, self).__init__()
        self.scaler = s
    def forward(self, x):
        return torch.tanh(self.scaler * x)


@gin.configurable
class BilateralGridCP4D(nn.Module):
    grid_X = 16  # Grid width.
    grid_Y = 16  # Grid height.
    grid_Z = 16  # Grid depth.
    grid_W = 8   # Guidance dimension.
    rank = 5     # Number of components
    learn_gray = True # If True, an MLP is trained to map RGB into guidance.
    gray_mlp_depth = 2 # Learnable guidance MLP depth.
    gray_mlp_width = 8 # Learnable guidance MLP width.
    init_noise_scale = 1e-6  # The noise scale of the initialized factors.
    bound = 2. # The scale of the bound.

    def __init__(self, **kwargs):
        super(BilateralGridCP4D, self).__init__()
        set_kwargs(self, kwargs)
        
        self.init_cp_factors_parafac()
        
        if self.learn_gray:
            rgb2gray_mlp_linear = lambda l: nn.Linear(self.gray_mlp_width, self.gray_mlp_width if l < self.gray_mlp_depth - 1 else 1)
            rgb2gray_mlp_actfn = lambda _: nn.ReLU(inplace=True)
            self.rgb2gray = nn.Sequential(
                *([nn.Linear(3, self.gray_mlp_width)] + \
                  [nn_module(l) for l in range(1, self.gray_mlp_depth) for nn_module in [rgb2gray_mlp_actfn, rgb2gray_mlp_linear]] + \
                  [_ScaledTanh(2.)]))
        else:
            # Weights of BT601/BT470 RGB-to-gray.
            self.register_buffer('rgb2gray_weight', torch.Tensor([[0.299, 0.587, 0.114]]))
            self.rgb2gray = lambda rgb: (rgb @ self.rgb2gray_weight.T) * 2. - 1.

    def _init_identity_grid(self):
        grid = torch.tensor([1., 0, 0, 0, 0, 1., 0, 0, 0, 0, 1., 0,]).float()
        grid = grid.repeat([self.grid_W * self.grid_Z * self.grid_Y * self.grid_X, 1]) 
        grid = grid.reshape(self.grid_W, self.grid_Z, self.grid_Y, self.grid_X, -1)
        grid = grid.permute(4, 0, 1, 2, 3)  # (12, grid_W, grid_Z, grid_Y, grid_X)
        return grid
    
    def init_cp_factors_parafac(self):
        # Initialize identity grids.
        init_grids = self._init_identity_grid()
        # Random noises are added to avoid singularity.
        init_grids = torch.randn_like(init_grids) * self.init_noise_scale + init_grids
        # Initialize grid CP factors
        _, facs = parafac(init_grids.clone().detach(), rank=self.rank)

        self.num_facs = len(facs)

        self.fac_0 = nn.Linear(facs[0].shape[0], facs[0].shape[1], bias=False)
        self.fac_0.weight = nn.Parameter(facs[0]) # (12, rank)

        for i in range(1, self.num_facs):
            fac = facs[i].T  # (rank, grid_size)
            fac = fac.view(1, fac.shape[0], fac.shape[1], 1) # (1, rank, grid_size, 1)
            self.register_buffer(f'fac_{i}_init', fac)

            fac_resid = torch.zeros_like(fac)
            self.register_parameter(f'fac_{i}', nn.Parameter(fac_resid))

    def forward(self, xyz, rgb):
        """
        :param xyz: (..., 3)
        :param rgb: (..., 3)
        :return: (..., 3, 4)
        """
        sh_ = xyz.shape
        xyz = xyz.reshape(-1, 3) # flatten (N, 3)
        rgb = rgb.reshape(-1, 3) # flatten (N, 3)

        xyz = xyz / self.bound

        gray = self.rgb2gray(rgb)
        xyzw = torch.cat([xyz, gray], dim=-1) # (N, 4)
        xyzw = xyzw.transpose(0, 1) # (4, N)
        coords = torch.stack([torch.zeros_like(xyzw), xyzw], dim=-1) # (4, N, 2)
        coords = coords.unsqueeze(1) # (4, 1, N, 2)        

        coef = 1.
        for i in range(1, self.num_facs):
            fac = self.get_parameter(f'fac_{i}') + self.get_buffer(f'fac_{i}_init')
            coef = coef * F.grid_sample(fac, coords[[i - 1]],
                                        align_corners=True,
                                        padding_mode='border') # [1, rank, 1, N]
        coef = coef.squeeze([0, 2]).transpose(0, 1) # (N, rank)
        mat = self.fac_0(coef)
        return mat.reshape(*sh_[:-1], 3, 4)
