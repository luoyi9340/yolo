# -*- coding: utf-8 -*-  
'''
v4.preprocess预测

@author: luoyi
Created on 2021年3月6日
'''
import tensorflow as tf
import numpy as np
#    print时不用科学计数法表示
np.set_printoptions(suppress=True) 

import models.layer.v4.preprocess as pp
import data.dataset_cells as ds_cells
from utils.iou import iou_n2n_tf_ragged


batch_size=4
db = ds_cells.tensor_db(batch_size=batch_size)
i = 10
idx = 0
for x,y in db:
    if (idx < i): 
        idx += 1
        continue
    y_true = y[:, 0, :]
    break
    pass



num_scales, num_anchors, num_classes, H,W = 3, 3, 1, 6,15
threshold_liable_iou = 0.25
yolohard = tf.random.uniform(shape=(batch_size, H, W, num_anchors, num_classes + 5))


liable_cells, y_true_liables, num_objects = pp.takeout_liable_cells(yolohard, y_true, batch_size)
liable_anchors = pp.takeout_liable_anchors(liable_cells, 
                                           y_true_liables, 
                                           fmaps_shape=[6, 15], 
                                           num_classes=num_classes, 
                                           num_anchors=num_anchors, 
                                           threshold_liable_iou=threshold_liable_iou)


