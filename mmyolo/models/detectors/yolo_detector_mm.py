"""
Author: Sijie Hu
Date: 18/03/2024
Description: This script adapted from mmyolo.yolo_detector.py to support multimodal inputs.
"""

from typing import List, Tuple, Union

import torch
from torch import Tensor

from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.dist import get_world_size
from mmengine.logging import print_log, MMLogger

from .yolo_detector import YOLODetector

from mmyolo.registry import MODELS

@MODELS.register_module()
class YOLODetector_MM(YOLODetector):
    r"""Implementation of YOLO Series (two branches)"""

    def loss(self, batch_inputs: List[Tensor],
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (list(Tensor)): Input 2 images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses

    def extract_feat(self, batch_inputs: List[Tensor]) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (list(Tensor)): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(*batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def predict(self,
                batch_inputs: List[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (list(Tensor)): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: List[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (list(Tensor)): Inputs with shape (N, C, H, W).

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def init_weights(self):
        if self.init_cfg is not None:
            if self.init_cfg.get('type', None) == 'RGB_Pretrained':
                from mmengine.runner.checkpoint import _load_checkpoint, load_state_dict
                logger = MMLogger.get_instance('mmengine')

                pretrained = self.init_cfg.get('checkpoint')
                print_log(f"load model from: {pretrained}", logger=logger)
                checkpoint = _load_checkpoint(pretrained, logger=logger, map_location='cpu')
                state_dict = checkpoint['state_dict']

                print_log(f"update stem conv for disparity: `new stem`", logger=logger)

                disp_branch = dict()
                for name, param in state_dict.items():
                    if 'stem' in name:
                        new_name = name.replace('stem', 'disp_stem')
                        disp_branch.update({new_name: param})
                    if 'stage1' in name:
                        new_name = name.replace('stage1', 'disp_stage1')
                        disp_branch.update({new_name: param})
                state_dict.update(disp_branch)
                load_state_dict(self, state_dict, strict=False, logger=logger)