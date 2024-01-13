# -*- coding: utf-8 -*-
"""
Created on Sun May  7 12:12:01 2023

@author: Vishal
"""

import scipy.io as i

mat = i.loadmat('Annotations.mat')

for key in mat.keys():
    variable = mat[key]
    print(f"Variable name: {key}")