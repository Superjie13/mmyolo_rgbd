# To run this script, you need to first install mmdeploy_multi_input (https://github.com/Superjie13/mmdeploy_multi_input) and mmyolo_rgbd. Place the pretrained model checkpoint in the checkpoint folder.
# mmdeploy, mmyolo_rgbd, and the checkpoint folder should be in the same working directory.
WORKSPACE=$(pwd)
MMDEPLOY_DIR=${WORKSPACE}/mmdeploy_multi_input
DEPLOY_CFG_PATH=${WORKSPACE}/mmyolo_rgbd/configs/deploy/detection_onnxruntime_static_mm.py
MODEL_CFG_PATH=${WORKSPACE}/mmyolo_rgbd/configs/yolox/custom/yolox_s_mmyolo_airsim_drone_disp.py
MODEL_CHECKPOINT_PATH=${WORKSPACE}/checkpoint/det_model.pth 

python3 ${MMDEPLOY_DIR}/tools/deploy_mm.py \
    ${DEPLOY_CFG_PATH} \
    ${MODEL_CFG_PATH} \
    ${MODEL_CHECKPOINT_PATH} \
    "${WORKSPACE}/mmyolo_rgbd/data/rgb_00000.png, ${WORKSPACE}/mmyolo_rgbd/data/disp_00000.png" \
    --work-dir deploy_output_2 \
    --device cpu \
    --log-level INFO \
    --show \
    --dump-info
