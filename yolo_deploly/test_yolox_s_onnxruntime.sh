MMDEPLOY_DIR=/media/sijeihu/Sijie/NII_Proj/model_deploy/mmdeploy
DEPLOY_CFG_PATH=/media/sijeihu/Sijie/NII_Proj/model_deploy/mmyolo/configs/deploy/detection_onnxruntime_static_mm.py
MODEL_CFG_PATH=/media/sijeihu/Sijie/NII_Proj/model_deploy/mmyolo/configs/yolox/custom/yolox_s_mmyolo_airsim_drone_disp.py
MODEL_CHECKPOINT_PATH=/media/sijeihu/Sijie/NII_Proj/model_deploy/mmyolo/work_dirs/yolox_s_mmyolo_airsim_drone_disp/airsimdrone_bbox_mAP_epoch_35.pth

python3 ${MMDEPLOY_DIR}/tools/deploy.py \
    ${DEPLOY_CFG_PATH} \
    ${MODEL_CFG_PATH} \
    ${MODEL_CHECKPOINT_PATH} \
    "/media/sijeihu/Sijie/NII_Proj/model_deploy/mmyolo/data/AirSim_drone/val/0000/left/00000.png, /media/sijeihu/Sijie/NII_Proj/model_deploy/mmyolo/data/AirSim_drone/val/0000/disparity/00000.png" \
    --work-dir work_dir \
    --device cpu \
    --log-level INFO \
    --show \
    --dump-info
