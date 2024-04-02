MMDEPLOY_DIR=/media/sijeihu/Sijie/NII_Proj/model_deploy/mmdeploy
DEPLOY_CFG_PATH=/media/sijeihu/Sijie/NII_Proj/model_deploy/mmyolo/configs/deploy/detection_tensorrt_dynamic-192x192-960x960.py
MODEL_CFG_PATH=/media/sijeihu/Sijie/NII_Proj/model_deploy/mmyolo/configs/yolox/yolox_tiny_fast_8xb8-300e_coco.py
MODEL_CHECKPOINT_PATH=/media/sijeihu/Sijie/NII_Proj/model_deploy/mmyolo/yolo_deploly/yolox_tiny_8xb8-300e_coco_20220919_090908-0e40a6fc.pth

python3 ${MMDEPLOY_DIR}/tools/deploy.py \
    ${DEPLOY_CFG_PATH} \
    ${MODEL_CFG_PATH} \
    ${MODEL_CHECKPOINT_PATH} \
    demo/dog.jpg \
    --work-dir work_dir \
    --device cuda:0 \
    --log-level INFO \
    --show \
    --dump-info
