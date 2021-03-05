# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年2月18日
'''
import tensorflow as tf

print(tf.version.VERSION)


num_object, num_anchors, num_classes = 6, 3, 1
threshold_liable_iou = 0.9
#    模拟cells tensor(num_object, num_anchors, num_classes+5)
cells = tf.random.uniform(shape=(num_object, num_anchors, num_classes + 5))
#    模拟IoU
iou = tf.random.uniform(shape=(num_object, num_anchors))

idx_threshold = tf.where(iou > threshold_liable_iou)
print(idx_threshold)

idx_max_iou = tf.math.argmax(iou, axis=-1)
idx_max_iou = tf.cast(idx_max_iou, dtype=tf.int32)
idxB = tf.range(num_object)
idx_max = tf.stack([idxB, idx_max_iou], axis=-1)

liable_anchors = []
idx_liables = []
for o in range(num_object):
    idx = idx_threshold[idx_threshold[:,0] == o]
    liables = tf.gather_nd(cells, idx)
    liable_anchors.append(liables)
    idx_liables.append(idx)
    pass
print(liable_anchors)
print(idx_liables)

for liable_anchor, idx_liable in zip(liable_anchors, idx_liables):
    print(liable_anchor, idx_liable)
    print()
    pass

