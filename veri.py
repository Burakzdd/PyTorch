#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 01:18:21 2021

@author: burakzdd
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose


#veri setinden eğtilecek verileri indiriyoruz
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

#veri test edilecek verileri indiriyoruz
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
#yinelenebilir veri yükleyicideki her eleman 64 özellik ve etiketlik bir grup döndürecekti
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("X [N, C, H, W]'nin boyutu': ", X.shape)
    print("y'nin boyutu': ", y.shape, y.dtype)
    break


