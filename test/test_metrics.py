# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年2月27日
'''
import tensorflow as tf
import numpy as np
#    print时不用科学计数法表示
np.set_printoptions(suppress=True)  


from models.layer.v4.metrics import YoloMetricBox, YoloMetricConfidence, YoloMetricUnConfidence, YoloMetricClasses
from models.layer.v4.preprocess import takeout_liable_cells, takeout_liable_anchors, takeout_unliable_anchors


#    模拟数据
batch_size, fmaps_shape, num_scale, num_anchors, num_classes = 2, [12,30], 3, 2, 1
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
    y_gt_c = tf.random.uniform(shape=(num_scale, 6, 1), minval=0, maxval=num_classes+1, dtype=tf.int32)
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
 
#    拿负责预测的cell和负责预测的cell对应的y_true信息
liable_cells, y_true_liable, num_object = takeout_liable_cells(yolohard, y_true)
print('\n负责预测的cell')
print(liable_cells)
print('\n负责预测的cell对应的y_true')
print(y_true_liable)
  
#    拿负责预测的anchors
liable_anchors_list = takeout_liable_anchors(liable_cells, y_true_liable, fmaps_shape, num_classes, num_anchors)
print('\n负责预测的anchors')
#    拿不负责预测的anchors
unliable_anchors = takeout_unliable_anchors(y_true_liable, yolohard, num_anchors)
print('\n不负责预测的anchors')
print(unliable_anchors)
  

yolo_metric_box = YoloMetricBox()
yolo_metric_confidence = YoloMetricConfidence()
yolo_metric_unconfidence = YoloMetricUnConfidence()
yolo_metric_classes = YoloMetricClasses()

mae_box = yolo_metric_box.mae_anchorbox_gtbox(liable_anchors_list, num_classes=num_classes)
mae_confidence = yolo_metric_confidence.mae_confidence(liable_anchors_list, num_classes=num_classes)
mae_unconfidence = yolo_metric_unconfidence.mae_unconfidence(unliable_anchors)
mae_classes = yolo_metric_classes.classes_info(liable_anchors_list, num_classes=num_classes)
print()
print(mae_box)
print(mae_confidence)
print(mae_unconfidence)
print(mae_classes)



