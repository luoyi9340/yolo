# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年3月2日
'''
import tensorflow as tf


x = tf.concat([tf.ones(shape=(1, 2, 2, 1), dtype=tf.int32),
               tf.ones(shape=(1, 2, 2, 1), dtype=tf.int32)*2], axis=0)
crop_size = [4, 4]

B = x.shape[0]
if (B == None): B = 1

boxes = tf.repeat(tf.convert_to_tensor([[0,0, 1,1]], dtype=tf.float32), repeats=B, axis=0)
boxes_idx = tf.range(B)
print(boxes)
print(boxes_idx)

y = tf.image.crop_and_resize(x, 
                             boxes=boxes, 
                             box_indices=boxes_idx, 
                             crop_size=crop_size)

print(tf.squeeze(x))
print(tf.squeeze(y))