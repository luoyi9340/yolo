# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年2月24日
'''
import numpy as np

import utils.conf as conf


scales_set=conf.DATASET_CELLS.get_scales_set()
print(scales_set)
standard_scale=np.array([conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT])
print(standard_scale)

scales_set = np.repeat(np.expand_dims(standard_scale, axis=0), repeats=scales_set.shape[0], axis=0) / np.expand_dims(scales_set, axis=-1)
scales_set = np.ceil(scales_set)
print(scales_set)