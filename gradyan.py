#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 02:13:02 2021

@author: burakzdd
"""


import torch

x = torch.ones(5)  #alınan tensör
y = torch.zeros(3)  #beklenen çıktı 
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
#geri dönük fonksiyonun gradyanın hesaplanması
loss.backward()
print(w.grad)
print(b.grad)


#.requires ile tensörün özelliğine bakılır
z = torch.matmul(x, w)+b
print(z.requires_grad)
#geri dönük işlemler durudurluyor
with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

#.detach işlemi torch.no_grad ile aynı işlemi yapar
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
