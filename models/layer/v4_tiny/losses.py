# -*- coding: utf-8 -*-  
'''
yolov4 tiny的loss

@author: luoyi
Created on 2021年3月9日
'''
import tensorflow as tf

import utils.conf as conf
import utils.alphabet as alphabet
import utils.logger_factory as logf
from models.layer.commons.losses import YoloV4BaseLosses
from models.layer.commons.part import YoloHardRegister
from models.layer.commons.part import AnchorsRegister
from models.layer.commons.preporcess import takeout_liables, takeout_unliables, parse_idxBHW

#    YoloV4TinyLosses
class YoloV4TingLosses(YoloV4BaseLosses):

    def __init__(self,
                 loss_lamda_box=conf.V4.get_loss_lamda_box(),
                 loss_lamda_confidence=conf.V4.get_loss_lamda_confidence(),
                 loss_lamda_unconfidence=conf.V4.get_loss_lamda_unconfidence(),
                 loss_lamda_cls=conf.V4.get_loss_lamda_cls(),
                 
                 yolohard_register=YoloHardRegister.instance(),
                 anchors_register=AnchorsRegister.instance(),
                 
                 batch_size=conf.DATASET_CELLS.get_batch_size(),
                 num_anchors=conf.DATASET_CELLS.get_anchors_set().shape[1],
                 num_classes=len(alphabet.ALPHABET),
                 max_objects=conf.V4.get_max_objects(),
                 threshold_liable_iou=conf.V4.get_threshold_liable_iou(),
                 
                 **kwargs):
        super(YoloV4TingLosses, self).__init__(**kwargs)
        
        self._loss_lamda_box = loss_lamda_box
        self._loss_lamda_confidence = loss_lamda_confidence
        self._loss_lamda_unconfidence = loss_lamda_unconfidence
        self._loss_lamda_cls = loss_lamda_cls
        
        self._yolohard_register = yolohard_register
        self._anchors_register = anchors_register
        
        self._batch_size = batch_size
        self._num_anchors = num_anchors
        self._num_classes = num_classes
        self._max_objects = max_objects
        self._threshold_liable_iou = threshold_liable_iou
        
        pass
    
    def call(self, y_true, y_pred):
        '''
            @param y_true: tensor(batch_size, num_scale, 6, 2 + 5 + num_anchor * 3)
            @param y_pred: 用不到，x数据从yolohard_register中拿
        '''
        #    解析yolohard值并暂存
        yolohard1 = self._yolohard_register.get_yolohard1()
        y_true1 = y_true[:, 0, :]
        liable_idxBHW1, liable_num_objects1 = parse_idxBHW(y_true1, self._batch_size)
        liable_anchors1, liable_num_objects1 = takeout_liables(liable_idxBHW1, liable_num_objects1, yolohard1, y_true1, self._num_anchors, self._batch_size, self._num_classes, self._threshold_liable_iou)
        unliable_anchors1, unliable_num_objects1 = takeout_unliables(liable_idxBHW1, liable_num_objects1, yolohard1, y_true1, self._batch_size, self._num_anchors, self._num_classes)
        self._anchors_register.deposit_yolohard1(liable_anchors1, liable_num_objects1, unliable_anchors1, unliable_num_objects1)
        
        yolohard2 = self._yolohard_register.get_yolohard2()
        y_true2 = y_true[:, 1, :]
        liable_idxBHW2, liable_num_objects2 = parse_idxBHW(y_true2, self._batch_size)
        liable_anchors2, liable_num_objects2 = takeout_liables(liable_idxBHW2, liable_num_objects2, yolohard2, y_true2, self._num_anchors, self._batch_size, self._num_classes, self._threshold_liable_iou)
        unliable_anchors2, unliable_num_objects2 = takeout_unliables(liable_idxBHW2, liable_num_objects2, yolohard2, y_true2, self._batch_size, self._num_anchors, self._num_classes)
        self._anchors_register.deposit_yolohard2(liable_anchors2, liable_num_objects2, unliable_anchors2, unliable_num_objects2)
        
        #    计算loss
        loss_boxes1 = self.loss_boxes(liable_anchors1, liable_num_objects1, fmaps_shape=(yolohard1.shape[1], yolohard1.shape[2]), num_classes=self._num_classes)
        loss_confidence1 = self.loss_confidence(liable_anchors1, liable_num_objects1, fmaps_shape=(yolohard1.shape[1], yolohard1.shape[2]), num_classes=self._num_classes)
        loss_unconfidence1 = self.loss_unconfidence(unliable_anchors1, unliable_num_objects1, fmaps_shape=(yolohard1.shape[1], yolohard1.shape[2]))
        loss_cls1 = self.loss_cls(liable_anchors1, liable_num_objects1, fmaps_shape=(yolohard1.shape[1], yolohard1.shape[2]), num_classes=self._num_classes)
        
        loss_boxes2 = self.loss_boxes(liable_anchors2, liable_num_objects2, fmaps_shape=(yolohard2.shape[1], yolohard2.shape[2]), num_classes=self._num_classes)
        loss_confidence2 = self.loss_confidence(liable_anchors2, liable_num_objects2, fmaps_shape=(yolohard2.shape[1], yolohard2.shape[2]), num_classes=self._num_classes)
        loss_unconfidence2 = self.loss_unconfidence(unliable_anchors2, unliable_num_objects2, fmaps_shape=(yolohard2.shape[1], yolohard2.shape[2]))
        loss_cls2 = self.loss_cls(liable_anchors2, liable_num_objects2, fmaps_shape=(yolohard2.shape[1], yolohard2.shape[2]), num_classes=self._num_classes)
        
        loss_boxes = (loss_boxes1 + loss_boxes2) / 2
        loss_confidence = (loss_confidence1 + loss_confidence2) / 2
        loss_unconfidence = (loss_unconfidence1 + loss_unconfidence2) / 2
        loss_cls = (loss_cls1 + loss_cls2) / 2
        loss = self._loss_lamda_box * loss_boxes \
                + self._loss_lamda_confidence * loss_confidence \
                + self._loss_lamda_unconfidence * loss_unconfidence \
                + self._loss_lamda_cls * loss_cls
        
        tf.print('loss:', loss, output_stream=logf.get_logger_filepath('losses_v4'))
                
        return loss
    

    pass

