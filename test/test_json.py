# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年2月24日
'''
import json
import numpy as np


b = np.array([1,1], dtype=np.int64)
a = {'a':b}
print(json.dumps(a))
