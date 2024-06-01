#!/bin/bash

# CONFIG=configs/360.gin  # For 360 scenes.
CONFIG=configs/llff.gin  # For forward-facing scenes.
SCENE=strat
EXPERIMENT=llff/"$SCENE"  # Checkpoints, results, and logs will be saved to exp/${EXPERIMENT}.
DATA_ROOT=/pathto/datasets/bilarf_data/testscenes/
DATA_DIR="$DATA_ROOT"/"$SCENE"


# Evaluation
python eval.py --gin_configs=${CONFIG} \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'"
