# Bilateral Guided Radiance Field Processing

This repository contains the code release for SIGGRAPH (TOG) 2024 paper: "Bilateral Guided Radiance Field Processing" by Yuehao Wang, Chaoyi Wang, Bingchen Gong, and Tianfan Xue.

**[Project Page](https://bilarfpro.github.io/) / [Arxiv](https://arxiv.org/abs/2406.00448) / [Data](https://huggingface.co/datasets/Yuehao/bilarf_data)**


![teaser](https://github.com/yuehaowang/bilarf/assets/6317569/2d7770a4-c7e8-4986-888d-d84984b45450)


## lib_bilagrid.py

Hesitate to use this "multinerf-derived" codebase? No worries! Our method is supposed to plug and play for various NeRF backbones. We assemble all the essential code related to 3D/4D bilateral grids in a single [lib_bilagrid.py](lib_bilagrid.py). You can download this file and import it into your codebase. The essential dependencies to install are PyTorch, Numpy, and [tensorly](https://github.com/tensorly/tensorly).

Please check out the documentation for a quick start and overview of using this module. Additionally, we provide two Colab examples: one demonstrates how to optimize a bilateral grid to approximate camera ISP, and the other shows how to optimize a 4D bilateral grid to enhance a 3D volumetric object.

[![View Documentation](https://img.shields.io/badge/View-Documentations-blue?style=for-the-badge&logo=readthedocs)](https://bilarfpro.github.io/docs/)
[![Example 1: Camera ISP](https://img.shields.io/badge/Colab_Demo-Camera_ISP-orange?style=for-the-badge&logo=jupyter)](https://colab.research.google.com/drive/1tx2qKtsHH9deDDnParMWrChcsa9i7Prr?usp=sharing)
[![Example 2: 3D Enhancement](https://img.shields.io/badge/Colab_Demo-3D_Enhancement-orange?style=for-the-badge&logo=jupyter)](https://colab.research.google.com/drive/17YOjQqgWFT3QI1vysOIH494rMYtt_mHL?usp=sharing)


> [!TIP]
> You could run the two examples with CPU runtime!

---

Our implementation is based on [zipnerf-pytorch](https://github.com/SuLvXiangXin/zipnerf-pytorch). To use this codebase or reproduce our results, please follow the instructions below. 

## Setup

```bash
# 1. Clone the repo.

# 2. Make a conda environment.
conda create --name bilarf python=3.9
conda activate bilarf

# 3. Install requirements.
# You'll probably need to install PyTorch separately to support your GPUs.
pip install -r requirements.txt

# 4. Install other extensions.
pip install ./gridencoder

# 5. Install a specific cuda version of torch_scatter.
# see more details at https://github.com/rusty1s/pytorch_scatter
CUDA=cu117 # note: please specify your installed cuda version
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
```


## Dataset

We use the following datasets in our experiments and demo video.

- [BilaRF Dataset](#bilarf-dataset)

- [Mip-NeRF 360 Dataset](https://jonbarron.info/mipnerf360/)

- [RawNeRF Dataset](https://bmild.github.io/rawnerf/)

- [DL3DV-10K Dataset](https://dl3dv-10k.github.io/DL3DV-10K/)

### BilaRF dataset

This dataset contains our own captured nighttime scenes, synthetic data generated from [RawNeRF dataset](https://bmild.github.io/rawnerf/), and editing samples. We host our dataset on Huggingface:

[![BilaRF Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-xl.svg)](https://huggingface.co/datasets/Yuehao/bilarf_data)

The dataset follows the file structure of [NeRF LLFF data](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7) (forward-facing scenes). In addition, the editing samples are stored in the 'edits/' directory. We also provide 'ext_metadata.json' that can offer info about scenes. The data loader in this codebase currently supports the following two fields:
```js
{
      /** The `spiral_radius_scale` field specifies the radius of spiral camera path to
       ensure view synthesis is not out of the bound of the reconstructed scene. */
	"spiral_radius_scale": 0.5,

      /** For scenes synthesized from RawNeRF dataset, `no_factor_suffix` is set to `true`
       to suggest loading downsized training images directly from the 'images/' directory
       instead of 'images_X/', where X is specified by `Config.factor`. */
	"no_factor_suffix": true
}
```

### Preparing custom data
To use your own data, you can type the commands below (suppose your captured images are saved in 'my_dataset_dir/images'). More elaborate instructions can be found [here](https://github.com/google-research/multinerf#using-your-own-data).

```bash
DATA_DIR=my_dataset_dir
bash tools/local_colmap_and_resize.sh ${DATA_DIR}

# Visualize COLMAP output.
python tools/vis_colmap.py --input_model ${DATA_DIR}/sparse/0 --input_format .bin
```

## Running

We provide example scripts for training, rendering, evaluation, and finishing in 'scripts/' (you will need to change the dataset path and experiment name):

```bash
# Train and render test and camera path.
sh scripts/train_render.sh
# Evaluation
sh eval.sh
# Finishing
sh finishing.sh
```

All the experiment results and checkpoints will be saved to the 'exp/' directory.

We use [gin](https://github.com/google/gin-config) as our configuration framework. The default gin configs can be found in 'configs/'. You can use `--gin_configs='...'` to specify a '.gin' file and use `--gin_bindings='...'` to add/override configurations.

Below are the bindings we added for our proposed approach.


### Bindings for bilateral guided training

#### Enable per-view bilateral grids in NeRF training:

```python
Model.bilateral_grid = True
```

#### Bilateral grid

```python
BilateralGrid.grid_width = 16  # Grid width.
BilateralGrid.grid_height = 16  # Grid height.
BilateralGrid.grid_depth = 8  # Guidance dimension.

Config.bilgrid_tv_loss_mult = 10.  # TV loss weight.
```

#### Render training views

```python
Config.render_train = True
Model.bilateral_grid = True  # If True, render training views with bilateral grids applied.
```
> [!CAUTION]
>  When rendering test views or a camera path, please ensure `Model.bilateral_grid = False`.

The checkpoints and results of the training stage will be saved to 'exp/`Config.exp_name`/'.

### Bindings for bilateral guided finishing

#### Enable 4D bilateral grid

```python
Model.bilateral_grid4d = True
```

#### Specify target view editing

```python
Config.exp_name = 'expname'  # Specify a base NeRF model to perform finishing.
Config.ft_name = 'edit_1'  # Specify a label for the editing.
Config.ft_tgt_image = 'edit_color_path_011.png'  # Path to the edited view.
Config.ft_tgt_pose = 'path:11'  # Camera pose identifier for the edited view.
```

The checkpoints and results of the finishing stage will be saved to 'exp/`Config.exp_name`/ft/`Config.ft_name`' sub-directory.

Regarding the values of `Config.ft_tgt_pose`, we use a simple syntax: `split-name:index`, where `split-name` is one of 'train', 'test', 'all', and 'path', and `index` is the camera index in the specified split.

> [!NOTE]
> 1. The split name 'all' refers to all captured images in the 'images/' directory（train+test views).
>
> 2. Camera poses in the 'path' split depend on `Config.render_path_frames`. Thus, we need to make sure the edited view is synthesized using the same parameters of the render path.

#### Low-rank 4D bilateral grid

```python
BilateralGridCP4D.grid_X = 16  # Grid width.
BilateralGridCP4D.grid_Y = 16  # Grid height.
BilateralGridCP4D.grid_Z = 16  # Grid depth.
BilateralGridCP4D.grid_W = 8  # Guidance dimension.

BilateralGridCP4D.rank = 5  # Number of components


## The following bindings are rarely modified in our experiments.

BilateralGridCP4D.learn_gray = True  # If True, an MLP is trained to map RGB into guidance.
BilateralGridCP4D.gray_mlp_depth = 2  # Learnable guidance MLP depth.
BilateralGridCP4D.gray_mlp_width = 8  # Learnable guidance MLP width.

BilateralGridCP4D.init_noise_scale = 1e-6  # The noise scale of the initialized factors.
BilateralGridCP4D.bound = 2.  # The scale of the bound.

Config.bilgrid4d_tv_loss_mult = 1.  # TV loss weight.
```

#### Rendering with 4D bilateral grid

```python
Config.render_ft = True
```

This will make 'render.py' reload the model with an optimized 4D bilateral grid in the 'ft/`Config.ft_name`' sub-directory. By removing this binding, 'render.py' will render the NeRF model without applying any 3D finishing.


## Acknowledgements

- Thanks to the contributors of [zipnerf-pytorch](https://github.com/SuLvXiangXin/zipnerf-pytorch) for their PyTorch implementation of ZipNeRF!
- Thanks to [Lu Ling](https://github.com/LuLing06) for sharing their pretty drone videos! Check out the amazing [DL3DV-10K](https://github.com/DL3DV-10K/Dataset) dataset!


## Citation
```
@article{wang2024bilateral,
    title={Bilateral Guided Radiance Field Processing},
    author={Wang, Yuehao and Wang, Chaoyi and Gong, Bingchen and Xue, Tianfan},
    journal={Arxiv},
    year={2024}
}
```



