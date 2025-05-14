# bash train_dfa.sh 0 "beagle_dog(s1)" 520 525 16 dfa_v1.0

#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

GPU=$1
object_name=$2
index_from=$3
index_to=$4
cam_idx=$5
version=$6
echo $cam_idx

CUDA_VISIBLE_DEVICES=0 python train_gui.py \
    --source_path "/root/wlsgur4011/GESI/SC-GS/data/DFA_processed/${object_name}" \
    --model_path "outputs/dfa/${object_name}" \
    --deform_type node \
    --node_num 512 \
    --hyper_dim 8 \
    --is_blender \
    --eval \
    --gt_alpha_mask_as_scene_mask \
    --local_frame \
    --resolution 2 \
    --W 800 \
    --H 800 \
    --white_background \
    --idx_from $index_from \
    --idx_to $index_to \
    --cam_idx $cam_idx \

