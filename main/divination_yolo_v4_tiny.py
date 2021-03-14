# -*- coding: utf-8 -*-  
'''
yolo_v4-tiny模型 占卜结果

@author: luoyi
Created on 2021年3月13日
'''
import tensorflow as tf
import matplotlib.pyplot as plot

import utils.conf as conf
import utils.alphabet as alphabet
import data.dataset as ds
from models.yolo_v4_tiny import YoloV4Tiny
from models.layer.commons.part import YoloHardRegister


#    训练好的模型参数路径
model_weights_path = conf.V4.get_save_weights_dir() + '/YoloV4Tiny_18_2.11.h5'

#    加载模型
yolo_v4_tiny = YoloV4Tiny()
yolo_v4_tiny.load_model_weight(model_weights_path)
yolo_v4_tiny.trainable = False
yolo_v4_tiny.show_info()

#    加载要占卜的图片
img_dir = conf.DATASET_CELLS.get_in_test()
img_name = '8fc02c5a-e144-4650-bdee-b9e83ec8eb12.png'
img_path = img_dir + '/' + img_name
img, img_shape = ds.read_image_to_arr(img_path)
#    有个bug，训练的时候归一化代码写错了，这里将错就错吧。。。
#img_tf = ((tf.convert_to_tensor(img) / 255.) - 0.5) * 2.
img_tf = ((tf.convert_to_tensor(img) - 255.) - 0.5) / 2.

qualified_anchors = yolo_v4_tiny.divination(img_tf, yolohard_register=YoloHardRegister.instance(), img_shape=img_shape, threshold_liable_iou=conf.V4.get_threshold_liable_iou())
qualified_anchors = qualified_anchors.numpy()

#    图上展示出来
def show_anchors(img, anchors):
    #    在图上划出
    fig = plot.figure()
    ax = fig.add_subplot(1,1,1)
    
    #    打印索引
    idxV = anchors[:, 5]
    vcode = [alphabet.index_category(int(v)) for v in idxV]
    print(vcode)
    
    for anchor in anchors:
        lx = int(anchor[0])
        ly = int(anchor[1])
        w = int(anchor[2] - anchor[0])
        h = int(anchor[3] - anchor[1])
        print(lx, ly, w, h)
        #    绘制检测的cell
        rect_cell = plot.Rectangle((lx, ly), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect_cell)
        pass
    
    #    绘制图像
    X = img / 255.
    plot.imshow(X)
    plot.show()
    pass

show_anchors(img, qualified_anchors)


