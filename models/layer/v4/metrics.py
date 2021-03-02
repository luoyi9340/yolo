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
            @param liable_anchors: list [(liable_anchors ... batch_size ...]
                                    liable_anchors: tensor(物体个数, num_classes + 7 + 6)
                                                                num_classes: 每个分类得分
                                                                7: anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, anchor的[xl,yl, xr,yr]
                                                                6: gt的[xl,yl, xr,yr, relative_area, idxV]
        '''
        mae = []
        #    取anchor的[xl,yl, xr,yr]
        for anchors in liable_anchors:
            #    都是tensor(num_object, 4)
            anchor_boxes = anchors[:, num_classes+3:num_classes+7]
            gt_boxes = anchors[:, num_classes+7:num_classes+11]
            #    计算两个box各个坐标的mae tensor(num_object, 4)
            mae_boxes = tf.math.abs(anchor_boxes - gt_boxes)
            mae_boxes = tf.math.reduce_mean(mae_boxes, axis=0)
            mae.append(mae_boxes)
            pass
        return tf.convert_to_tensor(mae)
        
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
    
    #    计算anchors的还原的box与gt_box各个坐标的MAE
    def mae_confidence(self, liable_anchors, num_classes=len(alphabet.ALPHABET)):
        '''
            @param liable_anchors: list [(liable_anchors ... batch_size ...]
                                    liable_anchors: tensor(物体个数, num_classes + 7 + 6)
                                                                num_classes: 每个分类得分
                                                                7: anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, anchor的[xl,yl, xr,yr]
                                                                6: gt的[xl,yl, xr,yr, relative_area, idxV]
        '''
        mae = []
        #    取anchor的[xl,yl, xr,yr]
        for anchors in liable_anchors:
            #    都是tensor(num_object, )
            anchor_confidence = anchors[:, num_classes]
            gt_confidence = anchors[:, num_classes + 1]
            mae_boxes = tf.math.abs(anchor_confidence - gt_confidence)
            mae_boxes = tf.math.reduce_mean(mae_boxes)
            mae.append(mae_boxes)
            pass
        return tf.convert_to_tensor(mae)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        #    从anchors_register中拿缓存结果
        #    y_true_liable
        if (self._yolohard_idx == 1): _, _, liable_anchors, _ = self._anchors_register.get_yolohard1()
        if (self._yolohard_idx == 2): _, _, liable_anchors, _ = self._anchors_register.get_yolohard2()
        if (self._yolohard_idx == 3): _, _, liable_anchors, _ = self._anchors_register.get_yolohard3()
        
        mae = self.mae_confidence(liable_anchors)
        mae_mean = tf.math.reduce_mean(mae)
        self.mae.assign(mae_mean)
        
        tf.print('confidence_mae:', mae, ' yolohard', self._yolohard_idx, output_stream=logf.get_logger_filepath('metrics_v4'))
        tf.print('confidence_mae_mean:', mae_mean, ' yolohard', self._yolohard_idx, output_stream=logf.get_logger_filepath('metrics_v4'))
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
    
    #    计算anchors的还原的box与gt_box各个坐标的MAE
    def mae_unconfidence(self, unliable_anchors):
        '''
            @param unliable_anchors: tensor(batch_size, H, W, num_anchors)
        '''
        mae = []
        for anchors in unliable_anchors:
            #    都是tensor(num_object, )
            mae_boxes = tf.math.abs(anchors)
            mae_boxes = tf.math.reduce_mean(mae_boxes)
            mae.append(mae_boxes)
            pass
        return tf.convert_to_tensor(mae)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        #    从anchors_register中拿缓存结果
        #    y_true_liable
        if (self._yolohard_idx == 1): _, _, _, unliable_anchors = self._anchors_register.get_yolohard1()
        if (self._yolohard_idx == 2): _, _, _, unliable_anchors = self._anchors_register.get_yolohard2()
        if (self._yolohard_idx == 3): _, _, _, unliable_anchors = self._anchors_register.get_yolohard3()
        
        mae = self.mae_unconfidence(unliable_anchors)
        mae_mean = tf.math.reduce_mean(mae)
        self.mae.assign(mae_mean)
        
        tf.print('unconfidence_mae', mae, ' yolohard', self._yolohard_idx, output_stream=logf.get_logger_filepath('metrics_v4'))
        tf.print('unconfidence_mae_mean', mae_mean, ' yolohard', self._yolohard_idx, output_stream=logf.get_logger_filepath('metrics_v4'))
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
            @param liable_anchors: list [(liable_anchors ... batch_size ...]
                                    liable_anchors: tensor(物体个数, num_classes + 7 + 6)
                                                                num_classes: 每个分类得分
                                                                7: anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, anchor的[xl,yl, xr,yr]
                                                                6: gt的[xl,yl, xr,yr, relative_area, idxV]
            @return: tensor(num_object, 2)
                        2: T, TP
        '''
        info = []
        for anchors in liable_anchors:
            #    取分类得分 tensor(num_object, num_classes)
            anchors_classes = anchors[:, :num_classes]
            #    取真实分类 tensor(num_object, )
            gt_classes = anchors[:, num_classes + 12]
            #    检测预测分类 tensor(num_object, )
            anchors_classes = tf.math.argmax(anchors_classes, axis=-1)
            #    比较anchors_classes 与 gt_classes
            cls_res = tf.equal(tf.cast(anchors_classes, dtype=tf.int32), 
                               tf.cast(gt_classes, dtype=tf.int32))
            T = tf.math.count_nonzero(cls_res)
            TP = cls_res.shape[0]
            
            info.append(tf.stack([T, TP], axis=-1))
            pass
        return tf.convert_to_tensor(info)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        #    从anchors_register中拿缓存结果
        #    y_true_liable
        if (self._yolohard_idx == 1): _, _, liable_anchors, _ = self._anchors_register.get_yolohard1()
        if (self._yolohard_idx == 2): _, _, liable_anchors, _ = self._anchors_register.get_yolohard2()
        if (self._yolohard_idx == 3): _, _, liable_anchors, _ = self._anchors_register.get_yolohard3()
        
        infos = self.classes_info(liable_anchors)
        p = tf.math.reduce_sum(infos[:,0]) / tf.math.reduce_sum(infos[:,1])
        self.mae.assign(tf.cast(p, dtype=tf.float32))
        
        tf.print('classes_mae', infos, ' yolohard', self._yolohard_idx, output_stream=logf.get_logger_filepath('metrics_v4'))
        tf.print('classes_mae_mean', p, ' yolohard', self._yolohard_idx, output_stream=logf.get_logger_filepath('metrics_v4'))
        pass
    
    def result(self):
        return self.mae
    def reset_states(self):
        self.mae.assign(0.)
        pass
    
    pass
