#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 03:20:47 2016

@author: galib
"""

from PIL import Image
import glob
import os

#path = '/home/galib/Desktop/Imgnet_downloads/Ready for anno/sharl anno voc'
#path = '/home/galib/Desktop/Kaggle_li/NCFM/datasets/annotation-li-corrected_kag'

path = '/home/galib/py-R-FCN-intel/data/VOCdevkit0712/VOC0712/Annotations'

#path = '/home/galib/Desktop/Extra Augment/bet_all_anno'

files = os.listdir(path)

files_txt = [i for i in files if i.endswith('.xml')]

thefile = open('cervical_imagename_aug.txt', 'a')

for item in files_txt:
  thefile.write("%s\n" % item)