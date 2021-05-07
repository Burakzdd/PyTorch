#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 23:00:12 2021

@author: burakzdd
"""
#ilk olarak kütüphanler aktifleştirilir
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#kodun kullanılacağı cihaz (cpu/gpu) belirlenir.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
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

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
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

model = NeuralNetwork().to(device)
print(model)



input_image = torch.rand(3,28,28)
print(input_image.size())
#nn.Flatten
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())    

##nn.Linear
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
#nn.Relu
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")


#nn.Sequintial
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
print(logits)

#nn.Softmax
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(pred_probab)
