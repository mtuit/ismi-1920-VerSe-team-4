# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 16:07:19 2020

@author: Casper
"""


import SimpleITK as sitk

def main():
    itk_img = sitk.ReadImage("verse004.mha")
    itk_arr = sitk.GetArrayFromImage(itk_img)
    print(itk_arr.shape)

if __name__ == '__main__':
    main()
