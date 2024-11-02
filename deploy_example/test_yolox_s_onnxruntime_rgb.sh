# To run this script, you need to first install mmdeploy (https://github.com/Superjie13/mmdeploy_multi_input) and mmyolo. Place the pretrained model checkpoint in the checkpoint folder.
# mmdeploy, mmyolo, and the checkpoint folder should be in the same working directory.

MMDEPLOY_DIR=./mmdeploy
DEPLOY_CFG_PATH=./mmyolo/configs/deploy/detection_onnxruntime_static_partition.py
MODEL_CFG_PATH=./mmyolo/configs/yolox/custom/yolox_s_mmyolo_airsim_drone.py
MODEL_CHECKPOINT_PATH=./checkpoint/rgb_ocsort_detection.pth 

python3 ${MMDEPLOY_DIR}/tools/deploy.py \
    ${DEPLOY_CFG_PATH} \
    ${MODEL_CFG_PATH} \
    ${MODEL_CHECKPOINT_PATH} \
    "./mmyolo/data/rgb_00000.png, ./mmyolo/data/disp_00000.png" \
    --work-dir work_dir/new_train \
    --device cpu \
    --log-level INFO \
    --show \
    --dump-info
