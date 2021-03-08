# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年2月18日
'''
import tensorflow as tf
import numpy as np
#    print时不用科学计数法表示
np.set_printoptions(suppress=True) 

print(tf.version.VERSION)


class TestLayer(tf.keras.layers.Layer):
    def __init__(self,
                 **kwargs):
        super(TestLayer, self).__init__(**kwargs)
        pass
    
    def call(self, x1=None, x2=None, **kwargs):
        return x1 + x2
    pass

tl = TestLayer()
y = tl(x1=1, x2=2)
print(y)