# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年2月26日
'''
import data.dataset_cells as ds_cells


db = ds_cells.tensor_db(batch_size=2)
for x,y in db:
    print(y.shape)
    pass