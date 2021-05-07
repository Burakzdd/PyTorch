#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:28:55 2021

@author: burakzdd
"""


#kütüphaneler aktifleştirilir
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


class CustomImageDataset(Dataset):
#init fonksiyonu başlatılır
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
#len fonksiyonu başlatılır
    def __len__(self):
        return len(self.img_labels)