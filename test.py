# -*- coding: utf-8 -*-

"""
@Title   : 
@Time    : 2024/2/1 17:00
@Author  : Biophilia Wu
@Email   : BiophiliaSWDA@163.com
"""
import torch
a = torch.randn(4, 4)
b = torch.randn(4, 4)

print(a, b, torch.maximum(a, b))
