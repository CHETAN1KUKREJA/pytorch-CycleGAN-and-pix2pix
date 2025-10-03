"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""

import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import monai.transforms as transforms
from monai.transforms import RandSpatialCropSamplesd, CenterSpatialCropd


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot
        self.params = {
            'num_pool': 100, #This is likely related to a GAN (Generative Adversarial Network) architecture like CycleGAN, where it specifies the size of an image buffer to store previously generated images.
            'roi_size': [128,128,128], #This defines the size of the 3D patch (Region of Interest or ROI) that will be cropped from the full images. The model will be trained on these smaller 128x128x128 voxel cubes.
            'samples_per_image': 8, 
            'pixdim':(1,1,1), #This sets the target voxel spacing in millimeters. Medical images can have different resolutions, so this step resamples all images to have a consistent physical size per voxel
            'imgA_intensity_range': (-1000,1000), #This specifies the original intensity range for Image A (likely a CT scan, as these values correspond to Hounsfield Units).
            'imgB_intensity_range': (0,1500), #This is the original intensity range for Image B (likely an MRI scan).
        }

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_transform_3D(opt, params=None):
    """
    Returns a MONAI transform pipeline for 3D medical image processing.

    Args:
        mode (str): 'train' for training transforms with augmentation, 
                    'test' for deterministic validation/testing transforms.
        params (dict): A dictionary containing parameters like 'pixdim', 'roi_size', etc.
    
    Returns:
        monai.transforms.Compose: The composed transform pipeline.
    """
    keys = ['imgA', 'imgB']
    base_transforms = [
        transforms.LoadImaged(keys=keys),
        transforms.EnsureChannelFirstd(keys=keys),
        transforms.Spacingd(keys=keys, pixdim=params['pixdim'], mode=("bilinear", "bilinear")),
        transforms.ScaleIntensityRanged(keys=['imgA'], a_min=params['imgA_intensity_range'][0], a_max=params['imgA_intensity_range'][1], b_min=-1.0, b_max=1.0, clip=True),
        transforms.ScaleIntensityRanged(keys=['imgB'], a_min=params['imgB_intensity_range'][0], a_max=params['imgB_intensity_range'][1], b_min=-1.0, b_max=1.0, clip=False),
        transforms.CropForegroundd(keys=keys, source_key='imgA'), # Crop both based on imgA's foreground
        transforms.SpatialPadd(keys=keys, spatial_size=params['roi_size'], method='end') # <-- MOVED HERE!
    ]

    if opt.isTrain:
        # Add random augmentations for training
        train_specific_transforms = [
            RandSpatialCropSamplesd(
                keys=keys,
                roi_size=params['roi_size'],
                num_samples=params['samples_per_image'],
                random_size=False,
                random_center=True
            )
        ]
        all_transforms = base_transforms + train_specific_transforms
    elif not opt.isTrain:
        # Use deterministic cropping for testing/validation
        test_specific_transforms = [
            transforms.CenterSpatialCropd(keys=keys, roi_size=params['roi_size']),
        ]
        all_transforms = base_transforms + test_specific_transforms
    else:
        raise ValueError(f"Mode '{opt.isTrain}' not recognized. Use 'True' or 'False'.")

    # SpatialPad is useful if cropping might result in a smaller-than-ROI size
    all_transforms.append(transforms.SpatialPadd(keys=keys, spatial_size=params['roi_size']))
    
    return transforms.Compose(all_transforms)