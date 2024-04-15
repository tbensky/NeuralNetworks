#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 21:17:15 2024

@author: tom
"""

Dw = 4
Dh = 4

Kw = 2
Kh = 2

def c2d(r,c,kernel_h,kernel_w):
    cs = [(r+1,c+1) for r in range(kernel_h) for c in range(kernel_w)]
    index = 0
    for i in range(r,r+kernel_h):
        for j in range(c,c+kernel_w):
            print(f"c{cs[index]} * d({i+1},{j+1})")
            index += 1

for kernel_row_place in range(0,Dh,Kh):
    for kernel_col_place in range(0,Dw,Kw):
        c2d(kernel_row_place,kernel_col_place,Kh,Kw)
        print("\n")