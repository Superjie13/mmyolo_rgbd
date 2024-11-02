_base_ = ['./base_static.py']
onnx_config = dict(
    input_names=['rgb', 'disp'],
    opset_version=11,)

# If you want to export part of the model, you can use partition_config to indicate the start and end of the model.
partition_config = dict(
    type='single_stage_mm',
    # Note: do not set the `apply_marks` in both `partition_config` and `codebase_config`, the
    # `apply_marks` in `codebase_config` will be ignored.
    apply_marks=True,  # set to True to apply marks for partition
    data_preprocessor=dict(
        type='DetDataPreprocessor_Disparity',
        pad_size_divisor=32,
        batch_augments=None
        ),
    partition_cfg=[
        dict(
            save_file='yolox.onnx',
            # The start node of the model, which is pre-defined in the model (mmyolo/deploy/models/detectors/yolo_detector_mm.py)
            start=['detector_forward:input'],
            # The end node of the model, which is pre-defined in the model (mmyolo/deploy/models/dense_heads/yolov5_head.py)
            end=['yolo_head_raw:input'],  # get scores and bboxes before nms
            output_names=[f'pred_maps.{i}' for i in range(3)])
    ])
codebase_config = dict(
    type='mmyolo',
    task='ObjectDetection_MM',
    model_type='end2end_mm',
    apply_marks=False,  # marks do not be supported in onnxruntime, so set to False
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1),
    module=['mmyolo.deploy'])
backend_config = dict(type='onnxruntime')


