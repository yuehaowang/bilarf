#!/bin/bash

# CONFIG=configs/360.gin  # For 360 scenes.
CONFIG=configs/llff.gin  # For forward-facing scenes.
SCENE=scibldg
EXPERIMENT=llff/"$SCENE"  # Checkpoints, results, logs will be saved to exp/${EXPERIMENT}.
DATA_ROOT=/pathto/datasets/bilarf_data/editscenes/
DATA_DIR="$DATA_ROOT"/"$SCENE"


# Training
# You can also run this with `accelerate launch`.
python train.py --gin_configs=${CONFIG} \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
    --gin_bindings="Model.bilateral_grid = True"


# Render testing views
python render.py --gin_configs=${CONFIG} \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'"


# Render path
python render.py --gin_configs=${CONFIG} \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
    --gin_bindings="Config.render_path = True" \
    --gin_bindings="Config.render_path_frames = 120" \
    --gin_bindings="Config.render_video_fps = 60"


# # Render training views
# # Comment the last line to render training views without
# # per-view bilateral grids applied.
# python render.py --gin_configs=${CONFIG} \
#     --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
#     --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
#     --gin_bindings="Config.render_train = True" \
#     --gin_bindings="Model.bilateral_grid = True"