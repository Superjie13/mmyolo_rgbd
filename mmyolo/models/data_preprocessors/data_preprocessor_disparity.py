"""
Author: Sijie Hu
Date: 18/03/2024
Description: This script adapted from mmdet.data_preprocessor.py to support disparity data.
"""

import math
from numbers import Number
from typing import Dict, List, Optional, Union, Sequence
import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F

from mmengine import MessageHub, is_list_of
from mmengine.utils import is_seq_of
from mmengine.model.utils import stack_batch
from mmdet.models.data_preprocessors import DetDataPreprocessor

from mmdet.models.utils.misc import samplelist_boxtype2tensor

from mmyolo.registry import MODELS


@MODELS.register_module()
class DetDataPreprocessor_Disparity(DetDataPreprocessor):
    """Image pre-process for multi-modal inputs.

    Comparing with the :class:`mmdet.models.DetDataPreprocessor`,

    1. This class supports multi-modal inputs, e.g., RGB and Disparity images
    2. This class does not support batch augmentation.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    """

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        batch_pad_shape = self._get_pad_shape({'inputs': data['inputs']['img']})
        data = self.cast_data(data)  # move data to the right device
        ori_inputs, data_samples = data['inputs'], data['data_samples']

        inputs = dict()
        for imgs_key, imgs in ori_inputs.items():
            _batch_inputs = imgs
            # Process data with `pseudo_collate`.
            if is_seq_of(_batch_inputs, torch.Tensor):
                batch_inputs = []
                for _batch_input in _batch_inputs:
                    # channel transform
                    if self._channel_conversion:
                        _batch_input = _batch_input[[2, 1, 0], ...]
                    # Convert to float after channel conversion to ensure
                    # efficiency
                    _batch_input = _batch_input.float()
                    # Normalization.
                    if self._enable_normalize:
                        if self.mean.shape[0] == 3:
                            assert _batch_input.dim(
                            ) == 3 and _batch_input.shape[0] == 3, (
                                'If the mean has 3 values, the input tensor '
                                'should in shape of (3, H, W), but got the tensor '
                                f'with shape {_batch_input.shape}')
                        _batch_input = (_batch_input - self.mean) / self.std
                    batch_inputs.append(_batch_input)
                # Pad and stack Tensor.
                batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor,
                                           self.pad_value)

            # Process data with `default_collate`.
            elif isinstance(_batch_inputs, torch.Tensor):
                assert _batch_inputs.dim() == 4, (
                    'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                    'or a list of tensor, but got a tensor with shape: '
                    f'{_batch_inputs.shape}')
                if self._channel_conversion:
                    _batch_inputs = _batch_inputs[:, [2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_inputs = _batch_inputs.float()
                if self._enable_normalize:
                    _batch_inputs = (_batch_inputs - self.mean) / self.std
                h, w = _batch_inputs.shape[2:]
                target_h = math.ceil(
                    h / self.pad_size_divisor) * self.pad_size_divisor
                target_w = math.ceil(
                    w / self.pad_size_divisor) * self.pad_size_divisor
                pad_h = target_h - h
                pad_w = target_w - w
                batch_inputs = F.pad(_batch_inputs, (0, pad_w, 0, pad_h),
                                     'constant', self.pad_value)
            else:
                raise TypeError('Output of `cast_data` should be a dict of '
                                'list/tuple with inputs and data_samples, '
                                f'but got {type(data)}： {data}')

            inputs[imgs_key] = batch_inputs

        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            batch_input_shape = tuple(inputs[list(inputs.keys())[0]][0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })

            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)

            if self.pad_mask and training:
                self.pad_gt_masks(data_samples)

            if self.pad_seg and training:
                self.pad_gt_sem_seg(data_samples)

        if training and self.batch_augments is not None:
            raise NotImplementedError(
                'Batch augmentation is not supported in this class.')
            # for batch_aug in self.batch_augments:
            #     inputs, data_samples = batch_aug(inputs, data_samples)

        # Convert the inputs (dict) into a list
        _inputs = list(inputs.values())

        return {'inputs': _inputs, 'data_samples': data_samples}