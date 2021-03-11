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

import models.layer.v4_tiny.preprocess as pp
import data.dataset_cells as ds_cells


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

#    取负责预测的anchors信息
liable_anchors, liable_sum_objects = pp.takeout_liables(yolohard=yolohard, y_true=y_true, num_anchors=num_anchors, batch_size=batch_size, num_classes=num_classes)
print(liable_anchors.shape, liable_sum_objects)

#    取不负责预测的anchors信息
unliable_anchors, unliable_sum_objects = pp.takeout_unliables(yolohard=yolohard, y_true=y_true, batch_size=batch_size, num_anchors=num_anchors, num_classes=num_classes)
print(unliable_anchors.shape, unliable_sum_objects)


