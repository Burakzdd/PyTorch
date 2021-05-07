#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 04:17:12 2021

@author: burakzdd
"""

import torch
import numpy as np
#doğrudan veriden oluşan
data = [[6, 7],[6, 7]]
x_data = torch.tensor(data)
print(x_data)
#numpy dizisinden oluşan
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Rastgele Tensör: \n {rand_tensor} \n")
print(f"Birler Tensörü: \n {ones_tensor} \n")
print(f"Sıfırlar Tensörü: \n {zeros_tensor}")


tensor = torch.rand(6,7)

print(f"Tensör Boyutu: {tensor.shape}")
print(f"Tensörün veri tip: {tensor.dtype}")
print(f"Tensörün kullanıldığı cihaz: {tensor.device}")

x_rand = torch.rand_like(x_data, dtype=torch.float) 
# overrides the datatype of x_data
print(f"Rastele Tensör: \n {x_rand} \n")


tensor = torch.ones(3, 3)
print('Brinci satır: ',tensor[0])
print('Birinci Sütun: ', tensor[:, 0])
print('Sonuncu Sütun:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)

topla= tensor.sum()
topla_item = topla.item()
print(topla_item, type(topla_item))


print(tensor, "\n")
tensor.add_(67)
print(tensor)

t = torch.ones(4)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")


n = np.ones(4)
t = torch.from_numpy(n)

np.add(n, 66, out=n)
print(f"t: {t}")
print(f"n: {n}")