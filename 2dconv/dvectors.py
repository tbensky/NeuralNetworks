#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:44:32 2024

@author: tom
"""

I = 4
K = 2
stride = 1

for row in range(0,I-K+1,stride):
    for col in range(0,I-K+1,stride):
        d = []
        for krow in range(0,K):
            for kcol in range(0,K):
                d.append((row + krow,col + kcol))
        print(f"d{row}{col}={d}")
        
