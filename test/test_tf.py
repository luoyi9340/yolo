# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年2月18日
'''
import tensorflow as tf

print(tf.version.VERSION)


num_object, num_classes = 4, 10
a = tf.random.uniform(shape=(num_object, num_classes))
b = tf.random.uniform(shape=(num_object,), minval=0, maxval=num_classes, dtype=tf.int32)
print(b)

a = tf.cast(tf.math.argmax(a, axis=-1), dtype=tf.int32)
print(a)

res = tf.equal(a, b)
print(res)
T = tf.math.count_nonzero(res)
TP = res.shape[0]
print(tf.stack([T, TP], axis=-1))


