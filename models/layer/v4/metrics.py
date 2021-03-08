# -*- coding: utf-8 -*-  
'''
评价指标

    - box回归评价指标
        计算预测box的[xl,yl, xr,yr]与真实物体[xl,yl, xr,yl]的MAE
        
    - 负责预测置信度评价指标
        计算预测置信度与真实置信度(IoU)的MAE

    - 不负责预测置信度评价指标
        计算预测置信度与真实置信度(0)的MAE
    
    - 分类准确度评价指标
        计算真实分类结果的准确率: P = T / TP

@author: luoyi
Created on 2021年3月1日
'''
import tensorflow as tf

import utils.logger_factory as logf
import utils.alphabet as alphabet
from models.layer.v4.preprocess import AnchorsRegister


#    box回归评价指标
class YoloMetricBox(tf.keras.metrics.Metric):
    '''box回归评价指标
        step1：通过负责预测的anchor还原出box_anchor [xl,yl, xr,yr]
        step2：通过gt的位置信息还原出box_gt [xl,yl, xr,yr]
        step3：计算box_anchor 与 box_gt 4个坐标值的MAE
    '''
    def __init__(self,
                 name='metric_box',
                 yolohard_idx=1,
                 anchors_register=AnchorsRegister.instance(),
                 **kwargs):
        '''
            @param name: metrics名称
            @param yolohard_idx: yolohard的index。取值范围：[1 | 2| 3]
            @param anchors_register: yolohard解析结果暂存器
        '''
        super(YoloMetricBox, self).__init__(name=name, **kwargs)
        self._anchors_register = anchors_register
        self._yolohard_idx = yolohard_idx
        
        self.mae = self.add_weight(name='box_mae', initializer='zero', dtype=tf.float32)
        pass
    
    #    计算anchors的还原的box与gt_box各个坐标的MAE
    def mae_anchorbox_gtbox(self, liable_anchors, num_classes=len(alphabet.ALPHABET)):
        '''
            @param RaggedTensor(batch_size, num_object, num_liable, num_classes + 7 + 6)
                                            batch_size: 图片批量个数
                                            num_object: 实际物体数，每张图片可能不一样
                                            num_liable: 负责预测的anchor数，每个物体可能不一样(0,3]
                                            num_classes + 7 + 6: 分类数 + anchor信息 + gt信息
                                                num_classes: 各个分类得分
                                                7: anchor的[xl,yl, xr,yr], anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, 
                                                6: gt的[xl,yl, xr,yr, relative_area, idxV]
            @return: Tensor(4,)
        '''
        #    取anchor还原的box左上右下坐标    RaggedTensor(batch_size, num_object(不固定), num_liable(不固定), 4)
        anchors_boxes = liable_anchors[:, :, :, num_classes : num_classes + 4]
        #    取gt还原的box左上右下坐标    RaggedTensor(batch_size, num_object(不固定), num_liable(不固定), 4)
        gt_boxes = liable_anchors[:, :, :, num_classes + 7 : num_classes + 7 + 4]
        
        #    计算坐标之间的绝对值距离    RaggedTensor(batch_size, num_object(不固定), num_liable(不固定), 4)
        d = tf.math.abs(anchors_boxes - gt_boxes)
        #    求每一个坐标的平均    Tensor(4,)
        mae_d = tf.math.reduce_mean(d, axis=(0, 1,2))
        return mae_d
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        #    从anchors_register中拿缓存结果
        #    y_true_liable
        if (self._yolohard_idx == 1): _, _, liable_anchors, _ = self._anchors_register.get_yolohard1()
        if (self._yolohard_idx == 2): _, _, liable_anchors, _ = self._anchors_register.get_yolohard2()
        if (self._yolohard_idx == 3): _, _, liable_anchors, _ = self._anchors_register.get_yolohard3()
        
        mae = self.mae_anchorbox_gtbox(liable_anchors)
        mae_mean = tf.math.reduce_mean(mae)
        self.mae.assign(mae_mean)
        
        tf.print('----------------------------------------------------------------------------------------------------', output_stream=logf.get_logger_filepath('metrics_v4'))      
        tf.print('box_mae:', mae, ' yolohard', self._yolohard_idx, output_stream=logf.get_logger_filepath('metrics_v4'))
        tf.print('box_mae_mean:', mae_mean, ' yolohard', self._yolohard_idx, output_stream=logf.get_logger_filepath('metrics_v4'))
        pass
    
    def result(self):
        return self.mae
    def reset_states(self):
        self.mae.assign(0.)
        pass
    
    pass


#    confidence回归指标
class YoloMetricConfidence(tf.keras.metrics.Metric):
    '''confidence回归指标
        直接计算anchor与gt的mae
    '''
    def __init__(self,
                 name='metric_confidence',
                 yolohard_idx=1,
                 anchors_register=AnchorsRegister.instance(),
                 **kwargs):
        '''
            @param name: metrics名称
            @param yolohard_idx: yolohard的index。取值范围：[1 | 2| 3]
            @param anchors_register: yolohard解析结果暂存器
        '''
        super(YoloMetricConfidence, self).__init__(name=name, **kwargs)
        self._anchors_register = anchors_register
        self._yolohard_idx = yolohard_idx
        
        self.mae = self.add_weight(name='confidence_mae', initializer='zero', dtype=tf.float32)
        pass
    
    #    计算anchors的预测置信度与标记置信度之间的差的绝对值
    def mae_confidence(self, liable_anchors, num_classes=len(alphabet.ALPHABET)):
        '''
            @param RaggedTensor(batch_size, num_object, num_liable, num_classes + 7 + 6)
                                            batch_size: 图片批量个数
                                            num_object: 实际物体数，每张图片可能不一样
                                            num_liable: 负责预测的anchor数，每个物体可能不一样(0,3]
                                            num_classes + 7 + 6: 分类数 + anchor信息 + gt信息
                                                num_classes: 各个分类得分
                                                7: anchor的[xl,yl, xr,yr], anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, 
                                                6: gt的[xl,yl, xr,yr, relative_area, idxV]
            @return: Tensor(1,)
        '''
        #    取预测置信度    RaggedTensor(batch_size, num_object(不固定), num_liable(不固定), 1)
        confidence_prob = liable_anchors[:, :, :, num_classes + 4 : num_classes + 4 + 1]
        
        #    取标记置信度    RaggedTensor(batch_size, num_object(不固定), num_liable(不固定), 1)
        confidence_true = liable_anchors[:, :, :, num_classes + 5 : num_classes + 5 + 1]
        
        #    计算置信度差的绝对值    RaggedTensor(batch_size, num_object(不固定), num_liable(不固定), 1)
        d_confidence = tf.math.abs(confidence_prob - confidence_true)
        mae = tf.math.reduce_mean(d_confidence)
        
        return mae
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        #    从anchors_register中拿缓存结果
        #    y_true_liable
        if (self._yolohard_idx == 1): _, _, liable_anchors, _ = self._anchors_register.get_yolohard1()
        if (self._yolohard_idx == 2): _, _, liable_anchors, _ = self._anchors_register.get_yolohard2()
        if (self._yolohard_idx == 3): _, _, liable_anchors, _ = self._anchors_register.get_yolohard3()
        
        mae = self.mae_confidence(liable_anchors)
        self.mae.assign(mae)
        
        tf.print('confidence_mae:', mae, ' yolohard', self._yolohard_idx, output_stream=logf.get_logger_filepath('metrics_v4'))
        pass
    
    def result(self):
        return self.mae
    def reset_states(self):
        self.mae.assign(0.)
        pass
    
    pass


#    confidence回归指标
class YoloMetricUnConfidence(tf.keras.metrics.Metric):
    '''confidence回归指标
        直接计算anchor与gt的mae
    '''
    def __init__(self,
                 name='metric_unconfidence',
                 yolohard_idx=1,
                 anchors_register=AnchorsRegister.instance(),
                 **kwargs):
        '''
            @param name: metrics名称
            @param yolohard_idx: yolohard的index。取值范围：[1 | 2| 3]
            @param anchors_register: yolohard解析结果暂存器
        '''
        super(YoloMetricUnConfidence, self).__init__(name=name, **kwargs)
        self._anchors_register = anchors_register
        self._yolohard_idx = yolohard_idx
        
        self.mae = self.add_weight(name='unconfidence_mae', initializer='zero', dtype=tf.float32)
        pass
    
    #    计算anchor的预测置信度与标记置信度(其实都是0)的差的绝对值
    def mae_unconfidence(self, unliable_anchors):
        '''
            @param unliable_anchors: Tensor(batch_size, H, W, num_anchors)
            @return: Tensor(1,)
        '''
        d_unconfidence = tf.math.abs(unliable_anchors)
        mae = tf.math.reduce_mean(d_unconfidence)
        return mae
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        #    从anchors_register中拿缓存结果
        #    y_true_liable
        if (self._yolohard_idx == 1): _, _, _, unliable_anchors = self._anchors_register.get_yolohard1()
        if (self._yolohard_idx == 2): _, _, _, unliable_anchors = self._anchors_register.get_yolohard2()
        if (self._yolohard_idx == 3): _, _, _, unliable_anchors = self._anchors_register.get_yolohard3()
        
        mae = self.mae_unconfidence(unliable_anchors)
        self.mae.assign(mae)
        
        tf.print('unconfidence_mae', mae, ' yolohard', self._yolohard_idx, output_stream=logf.get_logger_filepath('metrics_v4'))
        pass
    
    def result(self):
        return self.mae
    def reset_states(self):
        self.mae.assign(0.)
        pass
    
    pass


#    分类评价指标
#    confidence回归指标
class YoloMetricClasses(tf.keras.metrics.Metric):
    '''confidence回归指标
        P = T / TP
    '''
    def __init__(self,
                 name='metric_classes',
                 yolohard_idx=1,
                 anchors_register=AnchorsRegister.instance(),
                 **kwargs):
        '''
            @param name: metrics名称
            @param yolohard_idx: yolohard的index。取值范围：[1 | 2| 3]
            @param anchors_register: yolohard解析结果暂存器
        '''
        super(YoloMetricClasses, self).__init__(name=name, **kwargs)
        self._anchors_register = anchors_register
        self._yolohard_idx = yolohard_idx
        
        self.mae = self.add_weight(name='classes_mae', initializer='zero', dtype=tf.float32)
        pass
    
    def classes_info(self, liable_anchors, num_classes=len(alphabet.ALPHABET)):
        '''
            @param liable_anchors RaggedTensor(batch_size, num_object, num_liable, num_classes + 7 + 6)
                                            batch_size: 图片批量个数
                                            num_object: 实际物体数，每张图片可能不一样
                                            num_liable: 负责预测的anchor数，每个物体可能不一样(0,3]
                                            num_classes + 7 + 6: 分类数 + anchor信息 + gt信息
                                                num_classes: 各个分类得分
                                                7: anchor的[xl,yl, xr,yr], anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, 
                                                6: gt的[xl,yl, xr,yr, relative_area, idxV]
            @return: T, TP
        '''
        #    取各个分类的预测得分    RaggedTensor(batch_size, num_object, num_liable, num_classes)
        cls_prob = liable_anchors[:, :, :, :num_classes]
        #    RaggedTensor不支持argmax。弯一道，先求出值最大的得分，在判断哪个索引与最大值相等
        cls_prob_max = tf.math.reduce_max(cls_prob, axis=-1)
        cls_prob_max = tf.expand_dims(cls_prob_max, axis=-1)
        idx_cls_prob_max = tf.where(cls_prob == cls_prob_max)       #    Tensor(batch_size * num_object * num_liable, 4)
        cls_prob = idx_cls_prob_max[:, 3]                   #    Tensor(batch_size * num_object * num_liable, 1)
        
        #    标记的分类索引        RaggedTensor(batch_size, num_object, num_liable, 1)
        cls_true = liable_anchors[:, :, :, num_classes + 7 + 5 : num_classes + 7 + 6]
        cls_true = cls_true.to_tensor(-1)
        cls_true = tf.gather_nd(cls_true, tf.where(cls_true > 0))   #    Tensor(batch_size * num_object * num_liable, 1)
        cls_true = tf.cast(cls_true, dtype=tf.int64)
        
        condition = cls_prob == cls_true
        T = tf.math.count_nonzero(condition)
        TP = cls_true.shape[0]
        
        return T, TP
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        #    从anchors_register中拿缓存结果
        #    y_true_liable
        if (self._yolohard_idx == 1): _, _, liable_anchors, _ = self._anchors_register.get_yolohard1()
        if (self._yolohard_idx == 2): _, _, liable_anchors, _ = self._anchors_register.get_yolohard2()
        if (self._yolohard_idx == 3): _, _, liable_anchors, _ = self._anchors_register.get_yolohard3()
        
        T, TP = self.classes_info(liable_anchors)
        p = T / TP
        self.mae.assign(tf.cast(p, dtype=tf.float32))
        
        tf.print('classes_acc', p, ' yolohard', self._yolohard_idx, output_stream=logf.get_logger_filepath('metrics_v4'))
        pass
    
    def result(self):
        return self.mae
    def reset_states(self):
        self.mae.assign(0.)
        pass
    
    pass
