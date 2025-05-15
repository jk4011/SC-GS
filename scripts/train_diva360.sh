# bash train_dfa.sh 0 "beagle_dog(s1)" 520 525 16 dfa_v1.0
# /root/wlsgur4011/GESI/SC-GS/data/Diva360_data/processed_data/penguin

#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

GPU=$1
object_name=$2
idx_from=$3
idx_to=$4
cam_idx=$5
version=$6

CUDA_VISIBLE_DEVICES=${GPU} python train_gui.py \
    --source_path data/Diva360/$object_name \
    --model_path "outputs/diva360/$object_name" \
    --deform_type node \
    --node_num 10000 \
    --hyper_dim 8 \
    --is_blender \
    --eval \
    --gt_alpha_mask_as_scene_mask \
    --local_frame \
    --resolution 2 \
    --W 800 \
    --H 800 \
    --white_background \
    --idx_from $idx_from \
    --idx_to $idx_to \
    --cam_idx $cam_idx \
    --wandb_group $version \
    --is_diva360 \

##                                       GPU object             idx_from idx_to cam_idx wandb_group
# bash scripts/finetuning_drag_diva360.sh 5   penguin            0217     0239   00      tmp

# bash scripts/train_diva360.sh 7 penguin 0218 0239 00 tmp