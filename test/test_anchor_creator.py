# -*- coding: utf-8 -*-  
'''
先验anchor_creator测试

@author: luoyi
Created on 2021年2月23日
'''
import matplotlib.pyplot as plot
import numpy as np
import json

import utils.conf as conf
from data.dataset_anchor import AnchorCreator
from data.dataset import data_iterator


#    计算anchor代表
di = data_iterator(img_dir=conf.DATASET.get_in_train(),
                   label_path=conf.DATASET.get_label_train(),
                   is_mutiple_file=conf.DATASET.get_label_train_mutiple(),
                   count=conf.DATASET.get_count_train(),
                   )
asc = AnchorCreator(k=9, show_log=False)
res = asc.election(data_iterator=di, num_sample=100)
whs = asc.sort_split_hw(new_shape=(3,3, 5), centroids=res)
print(json.dumps(whs.tolist()))

#    在图上划出
fig = plot.figure()
ax = fig.add_subplot(1,1,1)

for a in res:
    xl = a[0]
    yl = a[1]
    xr = a[2]
    yr = a[3]
    rect = plot.Rectangle(((xl + xr)/2, (yl + yr)/2), (xr - xl), (yr - yl), fill=False, edgecolor='green',linewidth=1)
    ax.add_patch(rect)
    pass

X = np.ones(shape=(conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT, 3))
plot.imshow(X)
plot.show()