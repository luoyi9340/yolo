# -*- coding: utf-8 -*-  
'''
测试已经训练好的模型

@author: luoyi
Created on 2021年3月13日
'''
import tensorflow as tf

import utils.conf as conf
import utils.alphabet as alphabet
import data.dataset_cells as ds_cells
from models.yolo_v4_tiny import YoloV4Tiny
from models.layer.commons.part import YoloHardRegister, AnchorsRegister


#    训练好的模型参数路径
model_weights_path = conf.V4.get_save_weights_dir() + '/YoloV4Tiny_18_2.11.h5'

#    加载模型
yolo_v4_tiny = YoloV4Tiny()
yolo_v4_tiny.load_model_weight(model_weights_path)
yolo_v4_tiny.trainable = False
yolo_v4_tiny.show_info()

count=conf.DATASET_CELLS.get_count_train()
#    数据可能有问题，只能用当初训练的batch_size个数据。多了准确率就不对
count=4
#    加载测试数据
test_db = ds_cells.tensor_db(img_dir=conf.DATASET_CELLS.get_in_test(), 
                             label_path=conf.DATASET_CELLS.get_label_test(), 
                             is_label_mutiple=conf.DATASET_CELLS.get_label_test_mutiple(), 
                             count=count, 
                             batch_size=count, 
                             epochs=conf.DATASET_CELLS.get_epochs(), 
                             shuffle_buffer_rate=conf.DATASET_CELLS.get_shuffle_buffer_rate(), 
                             x_preprocess=lambda x:((x - 255.) - 0.5) / 2, 
                             y_preprocess=None, 
                             num_scales=conf.DATASET_CELLS.get_scales_set().shape[0], 
                             num_anchors=conf.DATASET_CELLS.get_anchors_set().shape[1], 
                             max_objects=conf.V4.get_max_objects())

X_test = []
Y_test = []
for x,y in test_db:
    X_test.append(x)
    Y_test.append(y)
    pass
X_test = tf.concat(X_test, axis=0) if len(X_test) > 1 else X_test[0]
Y_test = tf.concat(Y_test, axis=0) if len(Y_test) > 1 else Y_test[0]

res = yolo_v4_tiny.test_metrics(X_test, Y_test, 
                                yolohard_register=YoloHardRegister.instance(), 
                                anchors_register=AnchorsRegister.instance(), 
                                num_anchors=conf.DATASET_CELLS.get_anchors_set().shape[1], 
                                num_classes=len(alphabet.ALPHABET), 
                                batch_size=X_test.shape[0], 
                                threshold_liable_iou=conf.V4.get_threshold_liable_iou())
print(res)

