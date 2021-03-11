# -*- coding: utf-8 -*-  
'''
yolo v4 tiny 评价函数

@author: luoyi
Created on 2021年3月10日
'''
import tensorflow as tf

import utils.alphabet as alphabet
from models.layer.commons.part import AnchorsRegister


#    Boxes评价
class YoloV4MetricsBoxes(tf.keras.metrics.Metric):

    def __init__(self,
                 name='YoloV4MetricsBoxes',
                 yolohard_scale_idx=1,
                 anchors_register=AnchorsRegister.instance(),
                 num_classes=len(alphabet.ALPHABET),
                 **kwargs):
        '''
            @param name: metrics名称
            @param yolohard_idx: yolohard的index。取值范围：[1 | 2| 3]
            @param anchors_register: yolohard解析结果暂存器
        '''
        super(YoloV4MetricsBoxes, self).__init__(name=name, **kwargs)
        
        self._yolohard_scale_idx = yolohard_scale_idx
        self._anchors_register = anchors_register
        
        self._num_classes = num_classes
        
        self.mae = self.add_weight(name='box_mae', initializer='zero', dtype=tf.float32)
        pass
    
    #    计算anchors_boxes的[lx,ly, rx,ry]坐标与gts_boxes的绝对值误差
    def mae_boxes(self, liable_anchors, num_classes=len(alphabet.ALPHABET)):
        '''
            @param liable_anchors: Tensor(sum_object, num_anchors, num_classes + 7 + 6)
                                            sum_object: 这批训练数据中负责预测的cell总数（与num_objects的划分对应）
                                            num_anchors: 每个cells有多少个anchor。配置决定，目前是3个
                                            num_classes + 7 + 6 + 1: 每个anchor信息
                                                num_classes: 各个分类得分
                                                7: anchor的[xl,yl, xr,yr], anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, 
                                                6: gt的[xl,yl, xr,yr, relative_area, idxV]
                                                1: 表示该位置anchor是否负责预测。0：不负责，1：负责
        '''
        #    取anchors_boxes    Tensor(sum_object, num_anchors, 4)
        anchors_boxes = liable_anchors[:, :, num_classes : num_classes + 4]
        #    取gts_boxes        Tensor(sum_object, num_anchors, 4)
        gts_boxes = liable_anchors[:, :, num_classes + 7 : num_classes + 7 + 4]
        
        #    计算绝对值误差        Tensor(sum_object, num_anchors, 4)
        mae_boxes = tf.math.abs(anchors_boxes - gts_boxes)
        #    Tensor(4, )
        mae_every_loc = tf.math.reduce_mean(mae_boxes, axis=(0,1))
        return mae_every_loc
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        if (self._yolohard_scale_idx == 1): liable_anchors, _, _, _ = self._anchors_register.get_yolohard1()
        if (self._yolohard_scale_idx == 2): liable_anchors, _, _, _ = self._anchors_register.get_yolohard2()
        if (self._yolohard_scale_idx == 3): liable_anchors, _, _, _ = self._anchors_register.get_yolohard3()
        
        mae_every_loc = self.mae_boxes(liable_anchors, self._num_classes)
        mae = tf.math.reduce_mean(mae_every_loc)
        self.mae.assign(mae)
        pass
    
    def result(self):
        return self.mae
    def reset_states(self):
        self.mae.assign(0.)
        pass
    pass


#    置信度评价
class YoloV4MetricsConfidence(tf.keras.metrics.Metric):

    def __init__(self,
                 name='YoloV4MetricsConfidence',
                 yolohard_scale_idx=1,
                 anchors_register=AnchorsRegister.instance(),
                 num_classes=len(alphabet.ALPHABET),
                 **kwargs):
        '''
            @param name: metrics名称
            @param yolohard_idx: yolohard的index。取值范围：[1 | 2| 3]
            @param anchors_register: yolohard解析结果暂存器
        '''
        super(YoloV4MetricsConfidence, self).__init__(name=name, **kwargs)
        
        self._yolohard_scale_idx = yolohard_scale_idx
        self._anchors_register = anchors_register
        
        self._num_classes = num_classes
        
        self.mae = self.add_weight(name='box_confidence', initializer='zero', dtype=tf.float32)
        pass
    
    #    计算anchors_confidence与gts_confidence的绝对值误差
    def mae_confidence(self, liable_anchors, num_classes=len(alphabet.ALPHABET)):
        '''
            @param liable_anchors: Tensor(sum_object, num_anchors, num_classes + 7 + 6)
                                            sum_object: 这批训练数据中负责预测的cell总数（与num_objects的划分对应）
                                            num_anchors: 每个cells有多少个anchor。配置决定，目前是3个
                                            num_classes + 7 + 6 + 1: 每个anchor信息
                                                num_classes: 各个分类得分
                                                7: anchor的[xl,yl, xr,yr], anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, 
                                                6: gt的[xl,yl, xr,yr, relative_area, idxV]
                                                1: 表示该位置anchor是否负责预测。0：不负责，1：负责
        '''
        #    取anchors_confidence    Tensor(sum_object, num_anchors, )
        anchors_confidence = liable_anchors[:, :, num_classes + 4]
        #    取gts_confidence        Tensor(sum_object, num_anchors, )
        gts_confidence = liable_anchors[:, :, num_classes + 5]
        
        #    计算绝对值误差        Tensor(sum_object, num_anchors, )
        mae_confidence = tf.math.abs(anchors_confidence - gts_confidence)
        mae = tf.math.reduce_mean(mae_confidence)
        return mae
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        if (self._yolohard_scale_idx == 1): liable_anchors, _, _, _ = self._anchors_register.get_yolohard1()
        if (self._yolohard_scale_idx == 2): liable_anchors, _, _, _ = self._anchors_register.get_yolohard2()
        if (self._yolohard_scale_idx == 3): liable_anchors, _, _, _ = self._anchors_register.get_yolohard3()
        
        mae = self.mae_confidence(liable_anchors, self._num_classes)
        self.mae.assign(mae)
        pass
    
    def result(self):
        return self.mae
    def reset_states(self):
        self.mae.assign(0.)
        pass
    pass


#    负样本置信度评价
class YoloV4MetricsUnConfidence(tf.keras.metrics.Metric):

    def __init__(self,
                 name='YoloV4MetricsUnConfidence',
                 yolohard_scale_idx=1,
                 anchors_register=AnchorsRegister.instance(),
                 num_classes=len(alphabet.ALPHABET),
                 **kwargs):
        '''
            @param name: metrics名称
            @param yolohard_idx: yolohard的index。取值范围：[1 | 2| 3]
            @param anchors_register: yolohard解析结果暂存器
        '''
        super(YoloV4MetricsUnConfidence, self).__init__(name=name, **kwargs)
        
        self._yolohard_scale_idx = yolohard_scale_idx
        self._anchors_register = anchors_register
        
        self._num_classes = num_classes
        
        self.mae = self.add_weight(name='box_unconfidence', initializer='zero', dtype=tf.float32)
        pass
    
    #    计算anchors_confidence与gts_confidence的绝对值误差
    def mae_unconfidence(self, unliable_anchors, num_classes=len(alphabet.ALPHABET)):
        '''
            @param unliable_anchors: Tensor(sum_unliable_cells, num_anchors, )
                                            sum_unliable: 本轮数据中所有不负责检测的cells总数
                                            num_anchors: 每个anchor的预测置信度
        '''
        #    计算绝对值误差        Tensor(sum_unliable_cells, num_anchors, )
        mae_unconfidence = tf.math.abs(unliable_anchors - 0)
        mae = tf.math.reduce_mean(mae_unconfidence)
        return mae
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        if (self._yolohard_scale_idx == 1): _, _, unliable_anchors, _ = self._anchors_register.get_yolohard1()
        if (self._yolohard_scale_idx == 2): _, _, unliable_anchors, _ = self._anchors_register.get_yolohard2()
        if (self._yolohard_scale_idx == 3): _, _, unliable_anchors, _ = self._anchors_register.get_yolohard3()
        
        mae = self.mae_unconfidence(unliable_anchors, self._num_classes)
        self.mae.assign(mae)
        pass
    
    def result(self):
        return self.mae
    def reset_states(self):
        self.mae.assign(0.)
        pass
    pass


#    分类准确度评价
class YoloV4MetricsClasses(tf.keras.metrics.Metric):

    def __init__(self,
                 name='YoloV4MetricsClasses',
                 yolohard_scale_idx=1,
                 anchors_register=AnchorsRegister.instance(),
                 num_classes=len(alphabet.ALPHABET),
                 **kwargs):
        '''
            @param name: metrics名称
            @param yolohard_idx: yolohard的index。取值范围：[1 | 2| 3]
            @param anchors_register: yolohard解析结果暂存器
        '''
        super(YoloV4MetricsClasses, self).__init__(name=name, **kwargs)
        
        self._yolohard_scale_idx = yolohard_scale_idx
        self._anchors_register = anchors_register
        
        self._num_classes = num_classes
        
        self.mae = self.add_weight(name='box_classes', initializer='zero', dtype=tf.float32)
        pass
    
    #    计算anchors_confidence与gts_confidence的绝对值误差
    def mae_classes(self, liable_anchors, liable_num_objects, num_classes=len(alphabet.ALPHABET)):
        '''
            @param liable_anchors: Tensor(sum_object, num_anchors, num_classes + 7 + 6)
                                            sum_object: 这批训练数据中负责预测的cell总数（与num_objects的划分对应）
                                            num_anchors: 每个cells有多少个anchor。配置决定，目前是3个
                                            num_classes + 7 + 6 + 1: 每个anchor信息
                                                num_classes: 各个分类得分
                                                7: anchor的[xl,yl, xr,yr], anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, 
                                                6: gt的[xl,yl, xr,yr, relative_area, idxV]
                                                1: 表示该位置anchor是否负责预测。0：不负责，1：负责
            @param liable_num_objects: Tensor(batch_size,)   每个batch实际含有的物体数，idxBHW第1个维度的划分: 
        '''
        #    取每个anchor各个分类得分，并取得分最高的作为分类预测    Tensor(sum_object, num_anchors,)
        anchors_classes_prob = liable_anchors[:, :, :num_classes]
        anchors_classes_prob = tf.math.argmax(anchors_classes_prob, axis=-1)
        #    取每个anchor对应的真实分类索引    Tensor(sum_object, num_anchors,)
        anchors_classes_true = liable_anchors[:, :, num_classes + 7 + 5]
        anchors_classes_true = tf.cast(anchors_classes_true, dtype=tf.int64)
        
        #    比较
        equal_res = tf.equal(anchors_classes_prob, anchors_classes_true)
        
        T = tf.math.count_nonzero(equal_res)
        TP = tf.math.reduce_sum(liable_num_objects)
        return T, TP
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        if (self._yolohard_scale_idx == 1): liable_anchors, liable_num_objects, _, _ = self._anchors_register.get_yolohard1()
        if (self._yolohard_scale_idx == 2): liable_anchors, liable_num_objects, _, _ = self._anchors_register.get_yolohard2()
        if (self._yolohard_scale_idx == 3): liable_anchors, liable_num_objects, _, _ = self._anchors_register.get_yolohard3()
        
        T, TP = self.mae_classes(liable_anchors, liable_num_objects, self._num_classes)
        acc = tf.cast(T / TP, dtype=tf.float32)
        self.mae.assign(acc)
        pass
    
    def result(self):
        return self.mae
    def reset_states(self):
        self.mae.assign(0.)
        pass
    pass

