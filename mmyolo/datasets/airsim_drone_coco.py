"""
Author: Sijie Hu
Date: 18/03/2024
Description: This script contains the implementation of the AirSimDroneCoco dataset.
"""
import copy
import os.path as osp
from typing import List, Union, Any

from .coco_disparity import CocoDispDataset

from mmyolo.registry import DATASETS


@DATASETS.register_module()
class AirSimDroneCoco(CocoDispDataset):
    """AirSim Drone dataset in COCO format."""

    METAINFO = {
        'CLASSES':
        ('drone',)
    }