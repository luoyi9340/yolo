# -*- coding: utf-8 -*-  
'''
yolo模型定义

@author: luoyi
Created on 2021年3月2日
'''
import tensorflow as tf

import utils.conf as conf
import utils.alphabet as alphabet
from models.abstract_model import AModel
from models.layer.commons.part import YoloHardRegister, AnchorsRegister
from models.layer.v4.losses import YoloLosses
from models.layer.commons.metrics import YoloV4MetricsBoxes, YoloV4MetricsConfidence, YoloV4MetricsUnConfidence, YoloV4MetricsClasses
from models.layer.v4.layers import YoloV4Layer


#    YoloV4模型
class YoloV4(AModel):
    def __init__(self, 
                 learning_rate=0.01, 
                 
                 loss_lamda_box=conf.V4.get_loss_lamda_box(),
                 loss_lamda_confidence=conf.V4.get_loss_lamda_confidence(),
                 loss_lamda_unconfidence=conf.V4.get_loss_lamda_unconfidence(),
                 loss_lamda_cls=conf.V4.get_loss_lamda_cls(),
                 
                 num_anchors=conf.DATASET_CELLS.get_anchors_set().shape[1],
                 num_classes=len(alphabet.ALPHABET),
                 
                 yolohard_register=YoloHardRegister.instance(),
                 anchors_register=AnchorsRegister.instance(),
                 
                 input_shape=(None, conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT, 3),
                 is_build=True,
                 auto_assembling=True,
                 name="YoloV4"):
        
        self._learning_rate = learning_rate
        
        self._loss_lamda_box = loss_lamda_box
        self._loss_lamda_confidence = loss_lamda_confidence
        self._loss_lamda_unconfidence = loss_lamda_unconfidence
        self._loss_lamda_cls = loss_lamda_cls
        
        self._num_anchors = num_anchors
        self._num_classes = num_classes
        
        self._yolohard_register = yolohard_register
        self._anchors_register = anchors_register
        
        self._input_shape = input_shape
        self._name = name
        
        super(YoloV4, self).__init__(learning_rate, name, auto_assembling)
        
        if (is_build): self.build(input_shape=input_shape)
        pass
    
        #    优化器
    def create_optimizer(self, learning_rate=0.001):
        return tf.optimizers.Adam(learning_rate=learning_rate)
    #    损失函数
    def create_loss(self):
        return YoloLosses(yolohard_register=self._yolohard_register,
                          anchors_register=self._anchors_register,
                          loss_lamda_box=self._loss_lamda_box,
                          loss_lamda_confidence=self._loss_lamda_confidence,
                          loss_lamda_unconfidence=self._loss_lamda_unconfidence,
                          loss_lamda_cls=self._loss_lamda_cls
                          )
    #    评价函数
    def create_metrics(self):
        return [YoloV4MetricsBoxes(name='metrics_scale1_box', yolohard_scale_idx=1, anchors_register=self._anchors_register),
                YoloV4MetricsConfidence(name='metrics_scale1_confidence', yolohard_scale_idx=1, anchors_register=self._anchors_register),
                YoloV4MetricsUnConfidence(name='metrics_scale1_unconfidence', yolohard_scale_idx=1, anchors_register=self._anchors_register),
                YoloV4MetricsClasses(name='metrics_scale1_classes', yolohard_scale_idx=1, anchors_register=self._anchors_register),
                 
                YoloV4MetricsBoxes(name='metrics_scale2_box', yolohard_scale_idx=2, anchors_register=self._anchors_register),
                YoloV4MetricsConfidence(name='metrics_scale2_confidence', yolohard_scale_idx=2, anchors_register=self._anchors_register),
                YoloV4MetricsUnConfidence(name='metrics_scale2_unconfidence', yolohard_scale_idx=2, anchors_register=self._anchors_register),
                YoloV4MetricsClasses(name='metrics_scale2_classes', yolohard_scale_idx=2, anchors_register=self._anchors_register),
                 
                YoloV4MetricsBoxes(name='metrics_scale3_box', yolohard_scale_idx=3, anchors_register=self._anchors_register),
                YoloV4MetricsConfidence(name='metrics_scale3_confidence', yolohard_scale_idx=3, anchors_register=self._anchors_register),
                YoloV4MetricsUnConfidence(name='metrics_scale3_unconfidence', yolohard_scale_idx=3, anchors_register=self._anchors_register),
                YoloV4MetricsClasses(name='metrics_scale3_classes', yolohard_scale_idx=3, anchors_register=self._anchors_register)]
    #    模型名称
    def model_name(self):
        return self.name
    
        #    装配模型
    def assembling(self):
        #    创建yolo v4 layer
        self._layer = YoloV4Layer(name='yolo_v4_layer',
                                  input_shape=(self._input_shape[1], self._input_shape[2], self._input_shape[3]),
                                  num_anchors=self._num_anchors,
                                  num_classes=self._num_classes,
                                  yolohard_register=self._yolohard_register)
        pass
    
    def call(self, x, training=None, mask=None):
        y = self._layer(x)
        return y
    pass


