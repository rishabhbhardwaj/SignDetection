#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 21:49:36 2018

@author: rishabhbhardwaj
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class GermanTrafficSignDataset(Dataset):
    """German Traffic Sign dataset."""

    def __init__(self, gt_file, root_dir, transform=None):
        """
        Args:
            gt_file (string): Path to the grount truth file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.ground_truth = pd.read_csv(gt_file, sep=";", header = None)
        self.root_dir = root_dir
        self.transform = transform
        self.C = 3
        self.W = 1360
        self.H = 800

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.ground_truth.iloc[idx, 0])
        image = io.imread(img_name)
        #image = np.asarray(image).astype(np.float32)
        #image = image.reshape((self.C,self.Wself.H))
        #image = torch.from_numpy(image)
        #print("before shape", image.shape)
        #mage = image.view((self.C, self.H, self.W))
        #image = image.transpose((2, 0, 1))
        roi_points = self.ground_truth.iloc[idx, 1:-1].values.astype(np.float32)
        class_id = self.ground_truth.iloc[idx, -1].astype(np.float32)
        sample = {'img': image, 'roi': roi_points, 'class_id': class_id}

        if self.transform:
            image = self.transform(image)

        return {'img': image, 'roi': roi_points, 'class_id': class_id}
    
