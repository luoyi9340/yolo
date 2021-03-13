# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年2月27日
'''
import tensorflow as tf
import numpy as np
#    print时不用科学计数法表示
np.set_printoptions(suppress=True)  


from models.layer.v4_tiny.losses import YoloV4TingLosses
from models.layer.commons.preporcess import takeout_liables, takeout_unliables, parse_idxBHW


#    模拟数据
batch_size, fmaps_shape, num_scale, num_anchors, num_classes = 2, [12,30], 3, 2, 1
threshold_liable_iou = 0.5
#    模拟yolohard tensor(batch_size, 23, 60, num_anchors, num_classes+5)
yolohard = tf.random.uniform(shape=(batch_size, fmaps_shape[0], fmaps_shape[1], num_anchors, num_classes+5), dtype=tf.float32)
print(yolohard[0,0,0])
# print('yolohard')
# print(yolohard)
#    模拟y_true tensor(batch_size, num_scale, 6, 2 + 5 + num_anchor * 3)
y_true = []
for _ in range(batch_size):
#     y_xy = tf.convert_to_tensor([[0, 0]], dtype=tf.int32)
#     y_xy = tf.repeat(y_xy, repeats=6, axis=0)
    y_xy = tf.convert_to_tensor([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]], dtype=tf.int32)
    y_xy = tf.repeat(tf.expand_dims(y_xy, axis=0), repeats=num_scale, axis=0)
    y_gt_xywh = tf.random.uniform(shape=(num_scale, 6, 4))
    y_gt_c = tf.random.uniform(shape=(num_scale, 6, 1), minval=0, maxval=num_classes, dtype=tf.int32)
    y_anchor =  tf.random.uniform(shape=(num_scale, 6, num_anchors * 3))
    y = tf.concat([tf.cast(y_xy, dtype=tf.float32),
                   y_gt_xywh, tf.cast(y_gt_c, dtype=tf.float32),
                   y_anchor], axis=-1)
    y_true.append(y)
    pass
y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
#    取其中1个尺寸与yolo对应
y_true = y_true[:, 0, :,:]
print('\ny_true')
print(y_true)
 
 
liable_idxBHW, liable_num_objects = parse_idxBHW(y_true, batch_size)
#    拿负责预测的anchors
liable_anchors, liable_num_objects = takeout_liables(liable_idxBHW, liable_num_objects, yolohard, y_true, num_anchors, batch_size, num_classes, threshold_liable_iou)
print('\n负责预测的anchors')
print(liable_anchors)

#    拿不负责预测的anchors
unliable_anchors, unliable_sum_objects = takeout_unliables(liable_idxBHW, liable_num_objects, yolohard, y_true, batch_size, num_anchors, num_classes)
print('\n不负责预测的anchors')
print(unliable_anchors)
  
  
yolo_loss = YoloV4TingLosses()
loss_box = yolo_loss.loss_boxes(liable_anchors, liable_num_objects, fmaps_shape, num_classes)
loss_confidence = yolo_loss.loss_confidence(liable_anchors, liable_num_objects, fmaps_shape, num_classes)
loss_cls = yolo_loss.loss_cls(liable_anchors, liable_num_objects, fmaps_shape, num_anchors, num_classes)
loss_unconfidence = yolo_loss.loss_unconfidence(unliable_anchors, unliable_sum_objects, fmaps_shape)
print(loss_box)
print(loss_confidence)
print(loss_cls)
print(loss_unconfidence)


