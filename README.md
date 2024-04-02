## Multimodal MMYOLO
This repo is a multi-modal (rgb-d) version of [official mmyolo](README_mmyolo.md), which is specifically designed for 
the multimodal object detection task. Currently, we only test the multimodal YOLOX on our customized rgb-d (AirSim_Drone)
dataset.

To support the deployment of multimodal YOLOX, we also provide another repo [multimodal_mmdeploy]() for the deployment.

## Installation
You can install the multimodal mmyolo by following commands:
1. Create a conda environment
```shell
conda create -n mm_yolo python=3.8
conda activate mm_yolo
```
2. Install pytorch
```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
3. Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) 
using [MIM](https://github.com/open-mmlab/mim).
```shell
pip install -U openmim
mim install "mmengine=0.10.3"
mim install "mmcv=2.0.1"
mim install "mmdet=3.3.0"
```
4. Install the multimodal mmyolo
```shell
git clone git@github.com:Superjie13/multimodal_mmyolo.git
cd multimodal_mmyolo
# Install albumentations
pip install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

More details can be found in [official installation](https://mmyolo.readthedocs.io/en/latest/get_started/installation.html)

## Related 
- config
    1. __Training Related__
       - configs/yolox/custom/yolox_s_mmyolo_airsim_drone_disp.py
    2. __Deployment Related__
       - configs/deploy/detection_onnxruntime_static_mm.py (if you want to deploy the model, 
       you have to install [multimodal_mmdeploy]())

- mmyolo
    1. __Dataset Transform Related__
       - mmyolo/datasets/transforms/formatting_disparity.py
       - mmyolo/datasets/transforms/loading_disparity.py
       - mmyolo/datasets/transforms/mix_img_transforms_disparity.py
       - mmyolo/datasets/transforms/transforms_disparity.py
    2. __Dataset Related__
       - mmyolo/datasets/airsim_drone_coco.py
       - mmyolo/datasets/coco_disparity.py
    3. __Model Related__  
    __Note: torch.jit.trace does not support dict input, so the multimodel input should be a tuple of (img, disparity),
    instead of a dict of {"img": img, "disparity": disparity}__
       - ___backbone___
         - mmyolo/models/backbones/csp_darknet_mm.py
       - ___data_preprocessors___
         - mmyolo/models/data_preprocessors/data_preprocessor_disparity.py
       - ___detectors___
         - mmyolo/models/detectors/yolo_detector_mm.py
    4. __Deploy Related__
       - ___dense_heads___
         - mmyolo/deploy/models/dense_heads/yolov5_head.py
       - ___detectors___
         - mmyolo/deploy/models/detectors/yolo_detector_mm.py
       - mmyolo/deploy/object_detection_mm.py

## Usage
1. Train the multimodal yolox on AirSim_Drone dataset
```shell
python tools/train.py configs/yolox/custom/yolox_s_mmyolo_airsim_drone_disp.py
```
2. Deploy the multimodal yolox with onnxruntime
```shell
python path_to_multimodal_mmdeploy/tools/deploy_mm.py  \
  configs/deploy/detection_onnxruntime_static_mm.py  \
  configs/yolox/custom/yolox_s_mmyolo_airsim_drone_disp.py  \
  [path_to_pretrained_model.pth]  \
  "demo/rgb_00000.png, demo/disp_00000.png"  \
  --work-dir work_dir  \
  --device cpu  \
  --log-level INFO  \
  --dump-info  
```

2.1. Deploy the part of the multimodal yolox
> If you want to deploy the part of the multimodal yolox, you can check the `partition_config` part in
> configs/deploy/detection_onnxruntime_static_mm.py

3. Deploy the multimodal yolox with TensorRT (TODO)