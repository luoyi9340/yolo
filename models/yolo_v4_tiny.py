# -*- coding: utf-8 -*-  
'''
yolo_v4_tiny

@author: luoyi
Created on 2021年3月10日
'''
import tensorflow as tf

import utils.conf as conf
import utils.alphabet as alphabet
from models.abstract_model import AModel
from models.layer.commons.metrics import YoloV4MetricsBoxes, YoloV4MetricsConfidence, YoloV4MetricsUnConfidence, YoloV4MetricsClasses
from models.layer.v4_tiny.losses import YoloV4TingLosses
from models.layer.v4_tiny.layers import YoloV4TinyLayer


#    YoloV4Tiny模型
class YoloV4Tiny(AModel):
    def __init__(self, 
                 learning_rate=0.001, 
                 
                 loss_lamda_box=conf.V4.get_loss_lamda_box(),
                 loss_lamda_confidence=conf.V4.get_loss_lamda_confidence(),
                 loss_lamda_unconfidence=conf.V4.get_loss_lamda_unconfidence(),
                 loss_lamda_cls=conf.V4.get_loss_lamda_cls(),
                 
                 num_anchors=conf.DATASET_CELLS.get_anchors_set().shape[1],
                 num_classes=len(alphabet.ALPHABET),
                 
                 input_shape=(None, conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT, 3),
                 batch_size=conf.DATASET_CELLS.get_batch_size(),
                 is_build=True,
                 name="YoloV4Tiny"):
        
        self._learning_rate = learning_rate
        
        self._loss_lamda_box = loss_lamda_box
        self._loss_lamda_confidence = loss_lamda_confidence
        self._loss_lamda_unconfidence = loss_lamda_unconfidence
        self._loss_lamda_cls = loss_lamda_cls
        
        self._num_anchors = num_anchors
        self._num_classes = num_classes
        self._batch_size= batch_size
        
        self._input_shape = input_shape
        self._name = name
        
        super(YoloV4Tiny, self).__init__(learning_rate, name)
        
        if (is_build): self.build(input_shape=input_shape)
        pass
    
    #    优化器
    def create_optimizer(self, learning_rate=0.001):
        return tf.optimizers.Adam(learning_rate=learning_rate)
    #    损失函数
    def create_loss(self):
        return YoloV4TingLosses(loss_lamda_box=self._loss_lamda_box,
                                loss_lamda_confidence=self._loss_lamda_confidence,
                                loss_lamda_unconfidence=self._loss_lamda_unconfidence,
                                loss_lamda_cls=self._loss_lamda_cls)
    #    评价函数
    def create_metrics(self):
        return [YoloV4MetricsBoxes(name='yolohard1_boxes', yolohard_scale_idx=1),
                YoloV4MetricsConfidence(name='yolohard1_confidence', yolohard_scale_idx=1),
                YoloV4MetricsUnConfidence(name='yolohard1_unconfidence', yolohard_scale_idx=1),
                YoloV4MetricsClasses(name='yolohard1_classes', yolohard_scale_idx=1),
                
                YoloV4MetricsBoxes(name='yolohard2_boxes', yolohard_scale_idx=2),
                YoloV4MetricsConfidence(name='yolohard2_confidence', yolohard_scale_idx=2),
                YoloV4MetricsUnConfidence(name='yolohard2_unconfidence', yolohard_scale_idx=2),
                YoloV4MetricsClasses(name='yolohard2_classes', yolohard_scale_idx=2)]
    #    模型名称
    def model_name(self):
        return self.name
    
    #    装配模型
    def assembling(self):
        #    创建yolo v4 layer
        self._layer = YoloV4TinyLayer(name='yolo_v4_layer',
                                      input_shape=self._input_shape[1:],
                                      num_anchors=self._num_anchors,
                                      num_classes=self._num_classes)
        pass
    
    #    定义前向传播
    def call(self, inputs, training=None, mask=None):
        y = self._layer(inputs)
        return y
    pass

