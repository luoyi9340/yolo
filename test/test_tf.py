# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年2月18日
'''
import tensorflow as tf
import numpy as np
#    print时不用科学计数法表示
np.set_printoptions(suppress=True) 

from models.layer.v4_tiny.preprocess import takeout_liables

print(tf.version.VERSION)


a = tf.range(10)
a = tf.RaggedTensor.from_row_lengths(a, row_lengths=[4, 6])
print(a)
a = tf.math.reduce_mean(a, axis=-1)
print(a)
