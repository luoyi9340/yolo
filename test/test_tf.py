# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年2月18日
'''
import tensorflow as tf

print(tf.version.VERSION)


a = tf.random.uniform(shape=(10, 5))
b = tf.random.uniform(shape=(10,), minval=0, maxval=5, dtype=tf.int32)
b = tf.one_hot(b, 5)
print(a)
print(b)

loss = b * -tf.math.log(a) + (1 - b) * -tf.math.log(1 - a)
loss = tf.math.reduce_sum(loss, axis=-1)
print(loss)

