#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 01:31:19 2021

@author: burakzdd

"""
#PyThorc ile Model Oluşturma 

import torch
from torch import nn

#Eğitim için GPU'da mı yoksa CPU'da mı çalışıldığı kontrol edilir.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("{} cihazı kullanılıyor".format(device))

# Model tanımlanır
class YapaySinirAgi(nn.Module):
    def __init__(self):
        super(YapaySinirAgi, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = YapaySinirAgi().to(device)
print(model)
torch.save(model.state_dict(), "model.pth")
print("PyTorch modeli model.pth olarak kaydedildi")