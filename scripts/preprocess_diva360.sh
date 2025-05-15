#!/bin/bash

# 원하는 구조로 심볼릭 링크를 생성하는 스크립트입니다.
# 사용법: bash make_multipleview_symlinks.sh <output_dir>
# 예시: bash make_multipleview_symlinks.sh

set -e

# if [ $# -ne 1 ]; then
#     echo "Usage: $0 <output_dir>"
#     exit 1
# fi

OUTPUT_DIR="/data2/wlsgur4011/GESI/SC-GS/data"
SRC_ROOT="/data/rvi/dataset/Diva360_data/processed_data"

mkdir -p "$OUTPUT_DIR/Diva360"

for object_name in $(ls "$SRC_ROOT"); do
    echo -e "Processing object diva360: $object_name\n"
    
    OBJ_SRC="$SRC_ROOT/$object_name/frames_1"
    OBJ_DST="$OUTPUT_DIR/Diva360/$object_name"
    mkdir -p "$OBJ_DST"
    for cam_folder in $(ls "$OBJ_SRC"); do
        CAM_SRC="$OBJ_SRC/$cam_folder"
        CAM_DST="$OBJ_DST/$cam_folder"
        mkdir -p "$CAM_DST"
        for img in "$CAM_SRC"/*.png; do
            img_name=$(basename "$img")
            frame_num=$(echo "$img_name" | sed 's/^000//g') # .png 포함 경로
            # echo "image : $img, link : $CAM_DST/frame_${frame_num}"
            ln -sf "$img" "$CAM_DST/frame_${frame_num}"
        done
    done

    TRAIN_JSONS="$SRC_ROOT/$object_name/transforms_train.json"
    TEST_JSONS="$SRC_ROOT/$object_name/transforms_test.json"

    TRAIN_JSOND="$OBJ_DST/transforms_train.json"
    TEST_JSOND="$OBJ_DST/transforms_test.json"
    
    MERGED_JSON="$OBJ_DST/transforms_merged.json"   

    # echo $TRAIN_JSOND $TEST_JSOND $MERGED_JSON
    cp "$TRAIN_JSONS" "$TRAIN_JSOND"
    cp "$TEST_JSONS" "$TEST_JSOND"
    
    # # 먼저 undist/ 제거
    sed -i 's|undist/||g' "$TRAIN_JSOND" "$TEST_JSOND"
    
    # 그 다음 파일명 패턴 변경 (000xxxxx.png -> frame_xxxxx.png)
    sed -i 's|/000\([0-9]\{5\}\.png\)|/frame_\1|g' "$TRAIN_JSOND" "$TEST_JSOND"

    jq -s '{
    frames: (.[0].frames + .[1].frames),
    aabb_scale: .[0].aabb_scale
    }' ${TRAIN_JSOND} ${TEST_JSOND} > ${MERGED_JSON}
done

echo "심볼릭 링크 생성 완료: $OUTPUT_DIR/multipleview"