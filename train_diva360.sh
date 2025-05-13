# cd /root/wlsgur4011/GESI/SC-GS && conda activate sc-gs

CUDA_VISIBLE_DEVICES=0 python train_gui.py \
    --source_path /root/wlsgur4011/GESI/SC-GS/data/Diva360_data/processed_data/penguin \
    --model_path outputs/diva360/penguin \
    --deform_type node \
    --node_num 512 \
    --hyper_dim 8 \
    --is_blender \
    --eval \
    --gt_alpha_mask_as_scene_mask \
    --local_frame \
    --resolution 2 \
    --W 800 \
    --H 800