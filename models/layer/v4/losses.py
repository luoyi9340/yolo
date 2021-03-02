# -*- coding: utf-8 -*-  
'''
YOLO V4 的loss

    loss = λ[box] * loss_box
            + λ[confidence] * loss_confidence
            + λ[unconfidence] * loss_unconfidence
            + λ[cls] * loss_cls
            
    loss_box负责计算box的位置损失：
            loss_box = ∑(i∈cell) ∑[j∈anchors] U[i,j] * (2 - Area(GT)) * CIoU(anchor[i,j], GT[i,j])
            U[i,j] = 1 当第i个cell的第j个anchor负责物体时
                     0 当第i个cell的第j个anchor不负责物体时
            Area(GT) = 标注框面积（取值(0,1)，可选的。平衡大框与小框的loss贡献度）
                        Area(GT) = GT(w) * GT(h)
                        GT(w) = 标注框相对整图个宽度占比
                        GT(h) = 标注框相对整图的高度占比
            anchor[i,j] = 第i个cell中第j个anchor（负责物体检测的anchor）
                          注：IoU计算时采用归一化后的值计算
                              anchor[cx,cy]是anchor中心点相对cell左上角坐标的偏移量。[0,1]之间
                              anchor[w,h]是实际宽高相对整图的占比。(0,1]之间
            GT[i,j] = 第i个cell中第j个anchor负责检测的物体
                      注：IoU计算采用归一化后的值计算
                            GT[x,y]是标注框中心点相对cell左上角坐标的偏移量。[0,1]之间
                            GT[w,h]是实际宽高相对整图的占比。(0,1]之间
                            
    loss_confidence负责计算负责预测的anchor的置信度损失（这部分与V3一致）：                            
            loss_confidence = ∑(i∈cells) ∑(j∈anchors) U[i,j] * [c_[i,j] * -log(c[i,j])]
            U[i,j] = 1 当第i个cell的第j个anchor负责物体时
                     0 当第i个cell的第j个anchor不负责物体时
            c_[i,j] = 第i个cell的第j个anchor负责物体的置信度
                        c_[i,j] = P(object) * IoU(anchor, box)
                                = 1 * IoU(anchor, box)        （当anchor负责物体时，P(object)为1）
            c[i,j] = 第i个cell的第j个anchor预测负责物体的置信度
                
    loss_unconfidence负责计算不负责预测anchor的置信度损失：
            loss_unconfidence = ∑(i∈cells) ∑(j∈anchors) Un[i,j] * [(1 - c_[i,j]) * -log(1 - c[i,j])]
            Un[i,j] = 1 当第i个cell的第j个anchor不负责物体时
                      0 当第i个cell的第j个anchor负责物体时
            c_[i,j] = 第i个cell的第j个anchor负责物体的置信度
                        c_[i,j] = P(object) * IoU(anchor, box)
                                = 1 * IoU(anchor, box)        （当anchor负责物体时，P(object)为1）
            c[i,j] = 第i个cell的第j个anchor预测负责物体的置信度
            注：该项实际是让anchor学习认识背景
            
    loss_cls负责计算负责预测anchor的分类损失：
            λ[cls] = 1
            loss_cls = ∑(i∈cells) ∑(j∈anchors) ∑(c∈类别集合) U[i,j] * [p_[i,j,c] * -log(p[i,j,c]) + (1 - p_[i,j,c]) * -log(1 - p[i,j,c])]
            U[i,j] = 1 当第i个cell的第j个anchor负责物体时
                     0 当第i个cell的第j个anchor不负责物体时
            p_[i,j,c] = 第i个cell中第j个anchor负责物体的实际从属分类概率（one_hot）
            p[i,j,c] = 第i个cell中第j个anchor预测物体属于第c类的概率


@author: luoyi
Created on 2021年2月26日
'''
import tensorflow as tf

import utils.conf as conf
import utils.logger_factory as logf
import utils.alphabet as alphabet
from models.layer.commons.part import YoloHardRegister
from models.layer.v4.preprocess import AnchorsRegister

#    YOLO V4的loss
class YoloLosses(tf.keras.losses.Loss):
    def __init__(self, 
                 yolohard_register=YoloHardRegister.instance(),
                 anchors_register=AnchorsRegister.instance(),
                 loss_lamda_box=conf.V4.get_loss_lamda_box(),
                 loss_lamda_confidence=conf.V4.get_loss_lamda_confidence(),
                 loss_lamda_unconfidence=conf.V4.get_loss_lamda_unconfidence(),
                 loss_lamda_cls=conf.V4.get_loss_lamda_cls(),
                 **kwargs):
        '''
            @param yolohard_register: yolohard结果暂存
            @param anchors_register: yolohard解析为anchors结果暂存
        '''
        super(YoloLosses, self).__init__(**kwargs)
        
        self._yolohard_register = yolohard_register
        self._anchors_register = anchors_register
        
        self._loss_lamda_box = loss_lamda_box
        self._loss_lamda_confidence = loss_lamda_confidence
        self._loss_lamda_unconfidence = loss_lamda_unconfidence
        self._loss_lamda_cls = loss_lamda_cls
        pass
    
    #    计算loss
    def call(self, y_true, y_pred):
        '''
            @param y_true: tensor(batch_size, num_scale, 6, 2 + 5 + num_anchor * 3)
            @param y_pred: 用不到，x数据从yolohard_register中拿
        '''
        #    计算需要用到的数据，并缓存(metrics里也要用到，没必要每次都算一遍)
        (liable_anchors1, unliable_anchors1), \
        (liable_anchors2, unliable_anchors2), \
        (liable_anchors3, unliable_anchors3) = self._anchors_register.parse_anchors_and_cache(yolohard_register=self._yolohard_register, 
                                                                                              y_true=y_true)
        
        #    计算3种尺寸的loss，并求平均
        #    计算loss_box
        loss_box1 = self.loss_box(liable_anchors1)
        loss_box2 = self.loss_box(liable_anchors2)
        loss_box3 = self.loss_box(liable_anchors3)
        loss_box = (loss_box1 + loss_box2 + loss_box3) / 3.
        tf.print('----------------------------------------------------------------------------------------------------', output_stream=logf.get_logger_filepath('losses_v4_box'))        
        tf.print('loss_box1:', loss_box1, output_stream=logf.get_logger_filepath('losses_v4_box'))
        tf.print('loss_box2:', loss_box2, output_stream=logf.get_logger_filepath('losses_v4_box'))
        tf.print('loss_box3:', loss_box3, output_stream=logf.get_logger_filepath('losses_v4_box'))
        tf.print('loss_box:', loss_box, output_stream=logf.get_logger_filepath('losses_v4_box'))
        #    计算loss_confidence
        loss_confidence1 = self.loss_confidence(liable_anchors1)
        loss_confidence2 = self.loss_confidence(liable_anchors2)
        loss_confidence3 = self.loss_confidence(liable_anchors3)
        loss_confidence = (loss_confidence1 + loss_confidence2 + loss_confidence3) / 3.
        tf.print('----------------------------------------------------------------------------------------------------', output_stream=logf.get_logger_filepath('losses_v4_confidence'))        
        tf.print('loss_confidence1:', loss_confidence1, output_stream=logf.get_logger_filepath('losses_v4_confidence'))
        tf.print('loss_confidence2:', loss_confidence2, output_stream=logf.get_logger_filepath('losses_v4_confidence'))
        tf.print('loss_confidence3:', loss_confidence3, output_stream=logf.get_logger_filepath('losses_v4_confidence'))
        tf.print('loss_confidence:', loss_confidence, output_stream=logf.get_logger_filepath('losses_v4_confidence'))
        #    计算loss_unconfidence
        loss_unconfidence1 = self.loss_unconfidence(unliable_anchors1)
        loss_unconfidence2 = self.loss_unconfidence(unliable_anchors2)
        loss_unconfidence3 = self.loss_unconfidence(unliable_anchors3)
        loss_unconfidence = (loss_unconfidence1 + loss_unconfidence2 + loss_unconfidence3) / 3.
        tf.print('----------------------------------------------------------------------------------------------------', output_stream=logf.get_logger_filepath('losses_v4_unconfidence'))        
        tf.print('loss_unconfidence1:', loss_unconfidence1, output_stream=logf.get_logger_filepath('losses_v4_unconfidence'))
        tf.print('loss_unconfidence2:', loss_unconfidence2, output_stream=logf.get_logger_filepath('losses_v4_unconfidence'))
        tf.print('loss_unconfidence3:', loss_unconfidence3, output_stream=logf.get_logger_filepath('losses_v4_unconfidence'))
        tf.print('loss_unconfidence:', loss_unconfidence, output_stream=logf.get_logger_filepath('losses_v4_unconfidence'))
        #    计算loss_cls
        loss_cls1 = self.loss_cls(liable_anchors1)
        loss_cls2 = self.loss_cls(liable_anchors2)
        loss_cls3 = self.loss_cls(liable_anchors3)
        loss_cls = (loss_cls1 + loss_cls2 + loss_cls3) / 3.
        tf.print('----------------------------------------------------------------------------------------------------', output_stream=logf.get_logger_filepath('losses_v4_classes'))        
        tf.print('loss_cls1:', loss_cls1, output_stream=logf.get_logger_filepath('losses_v4_classes'))
        tf.print('loss_cls2:', loss_cls2, output_stream=logf.get_logger_filepath('losses_v4_classes'))
        tf.print('loss_cls3:', loss_cls3, output_stream=logf.get_logger_filepath('losses_v4_classes'))
        tf.print('loss_cls:', loss_cls, output_stream=logf.get_logger_filepath('losses_v4_classes'))
        
        loss = self._loss_lamda_box * loss_box \
                + self._loss_lamda_confidence * loss_confidence \
                + self._loss_lamda_unconfidence * loss_unconfidence \
                + self._loss_lamda_cls * loss_cls
                
        tf.print('----------------------------------------------------------------------------------------------------', output_stream=logf.get_logger_filepath('losses_v4'))        
        tf.print('loss_box:', loss_box, output_stream=logf.get_logger_filepath('losses_v4'))
        tf.print('loss_confidence:', loss_confidence, output_stream=logf.get_logger_filepath('losses_v4'))
        tf.print('loss_unconfidence:', loss_unconfidence, output_stream=logf.get_logger_filepath('losses_v4'))
        tf.print('loss:', loss, output_stream=logf.get_logger_filepath('losses_v4'))
        
        return loss
    
    #    计算loss_box
    def loss_box(self, anchors, num_classes=len(alphabet.ALPHABET)):
        '''
            loss_box = ∑(i∈cell) ∑[j∈anchors] U[i,j] * (2 - Area(GT)) * CIoU(anchor[i,j], GT[i,j])
            U[i,j] = 1 当第i个cell的第j个anchor负责物体时
                     0 当第i个cell的第j个anchor不负责物体时
            Area(GT) = 标注框面积（取值(0,1)，可选的。平衡大框与小框的loss贡献度）
                        Area(GT) = GT(w) * GT(h)
                        GT(w) = 标注框相对整图个宽度占比
                        GT(h) = 标注框相对整图的高度占比
            anchor[i,j] = 第i个cell中第j个anchor（负责物体检测的anchor）
                          注：IoU计算时采用归一化后的值计算
                              anchor[cx,cy]是anchor中心点相对cell左上角坐标的偏移量。[0,1]之间
                              anchor[w,h]是实际宽高相对整图的占比。(0,1]之间
            GT[i,j] = 第i个cell中第j个anchor负责检测的物体
                      注：IoU计算采用归一化后的值计算
                            GT[x,y]是标注框中心点相对cell左上角坐标的偏移量。[0,1]之间
                            GT[w,h]是实际宽高相对整图的占比。(0,1]之间
                            
            @param anchors: list [liable_anchors, ... batch_size个 ...]
                                    liable_anchors: tensor(物体个数, num_classes + 7 + 4)
                                                    num_classes: 每个分类得分
                                                    7: anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, anchor的[xl,yl, xr,yr]
                                                    6: gt的[xl,yl, xr,yr, relative_area, idxV]
            @return: tensor(batch_size, 1)
        '''
        loss = []
        #    遍历每一个batch_size的数据，计算loss_box
        for liable_anchors in anchors:
            #    从liable_anchors中取area(GT), CIoU
            area_gt = liable_anchors[:, num_classes + 7 + 4]        #    tensor(num_object, )
            ciou = liable_anchors[:, num_classes + 2]               #    tensor(num_object, )
            
            #    计算loss_box = ∑(i∈cell) ∑[j∈anchors] U[i,j] * (2 - Area(GT)) * CIoU(anchor[i,j], GT[i,j])
            loss_box = (2 - area_gt) * ciou                         #    tensor(num_object, 1)
            loss_box = tf.math.reduce_mean(loss_box)
            loss.append(loss_box)
            pass
        
        loss = tf.convert_to_tensor(loss, dtype=tf.float32)
        return loss
    
    #    计算loss_confidence
    def loss_confidence(self, anchors, num_classes=len(alphabet.ALPHABET)):
        '''loss_confidence负责计算负责预测的anchor的置信度损失（这部分与V3一致）：                            
            loss_confidence = ∑(i∈cells) ∑(j∈anchors) U[i,j] * [c_[i,j] * -log(c[i,j])]
                U[i,j] = 1 当第i个cell的第j个anchor负责物体时
                         0 当第i个cell的第j个anchor不负责物体时
                c_[i,j] = 第i个cell的第j个anchor负责物体的置信度
                            c_[i,j] = P(object) * IoU(anchor, box)
                                    = 1 * IoU(anchor, box)        （当anchor负责物体时，P(object)为1）
                c[i,j] = 第i个cell的第j个anchor预测负责物体的置信度
            @params anchors: list [liable_anchors, ... batch_size个 ...]
                                    liable_anchors: tensor(物体个数, num_classes + 7 + 4)
                                                    num_classes: 每个分类得分
                                                    7: anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, anchor的[xl,yl, xr,yr]
                                                    6: gt的[xl,yl, xr,yr, relative_area, idxV]
            @return: tensor(batch_size, 1)
        '''
        #    循环每个batch_size
        loss = []
        for liable_anchors in anchors:
            #    取置信度预测和置信度标签
            confidence_prob = liable_anchors[:, num_classes]
            confidence_true = liable_anchors[:, num_classes + 1]
            #    loss_confidence = ∑(i∈cells) ∑(j∈anchors) U[i,j] * [c_[i,j] * -log(c[i,j])]
            loss_confidence = confidence_true * (- tf.math.log(confidence_prob))
            loss_confidence = tf.math.reduce_mean(loss_confidence)
            loss.append(loss_confidence)
            pass
        loss = tf.convert_to_tensor(loss)
        return loss
    
    #    计算loss_unconfidence
    def loss_unconfidence(self, anchors):
        '''loss_unconfidence负责计算不负责预测anchor的置信度损失：
            loss_unconfidence负责计算不负责预测anchor的置信度损失：
                loss_unconfidence = ∑(i∈cells) ∑(j∈anchors) Un[i,j] * [(1 - c_[i,j]) * -log(1 - c[i,j])]
                Un[i,j] = 1 当第i个cell的第j个anchor不负责物体时
                          0 当第i个cell的第j个anchor负责物体时
                c_[i,j] = 第i个cell的第j个anchor负责物体的置信度
                            c_[i,j] = P(object) * IoU(anchor, box)
                                    = 1 * IoU(anchor, box)        （当anchor负责物体时，P(object)为1）
                c[i,j] = 第i个cell的第j个anchor预测负责物体的置信度
                注：该项实际是让anchor学习认识背景
            @params anchors: tensor(batch_size, H, W, num_anchors)
            @return: tensor(batch_size, 1)
        '''
        loss = []
        #    把batch_size循环掉
        for anchor in anchors:
            anchor_liable = tf.gather_nd(anchor, indices=tf.where(anchor > 0))
            #    loss_unconfidence = ∑(i∈cells) ∑(j∈anchors) Un[i,j] * [(1 - c_[i,j]) * -log(1 - c[i,j])]
            #    不用纠结标签让他优化到多少了，都是0。(confidence = P(objecg) * IoU P(Object)=0)
            loss_unconfidence = - tf.math.log(1 - anchor_liable)
            loss_unconfidence = tf.reduce_mean(loss_unconfidence)
            loss.append(loss_unconfidence)
            pass
        loss = tf.convert_to_tensor(loss)
        return loss
    
    #    计算loss_cls
    def loss_cls(self, anchors, num_classes=len(alphabet.ALPHABET)):
        '''loss_cls负责计算负责预测anchor的分类损失：
            loss_cls负责计算负责预测anchor的分类损失：
                λ[cls] = 1
                loss_cls = ∑(i∈cells) ∑(j∈anchors) ∑(c∈类别集合) U[i,j] * [p_[i,j,c] * -log(p[i,j,c]) + (1 - p_[i,j,c]) * -log(1 - p[i,j,c])]
                U[i,j] = 1 当第i个cell的第j个anchor负责物体时
                         0 当第i个cell的第j个anchor不负责物体时
                p_[i,j,c] = 第i个cell中第j个anchor负责物体的实际从属分类概率（one_hot）
                p[i,j,c] = 第i个cell中第j个anchor预测物体属于第c类的概率
            @params anchors: list [liable_anchors, ... batch_size个 ...]
                                    liable_anchors: tensor(物体个数, num_classes + 7 + 6)
                                                    num_classes: 每个分类得分
                                                    7: anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, anchor的[xl,yl, xr,yr]
                                                    6: gt的[xl,yl, xr,yr, relative_area, idxV]
            @return: tensor(batch_size, 1)
        '''
                #    循环每个batch_size
        loss = []
        for liable_anchors in anchors:
            #    取分类预测 tensor(num_object, num_classes)
            cls_prob = liable_anchors[:, :num_classes]
            #    取真实分类索引 tensor(num_object, )
            cls_true = tf.cast(liable_anchors[:, num_classes + 7 + 5], dtype=tf.int64)
            
            #    通过cls_true从cls_prob里取需要用到的分类预测
            idx_cls = tf.stack([tf.range(cls_true.shape[0], dtype=tf.int64),
                                cls_true], axis=-1)
            cls_prob = tf.gather_nd(cls_prob, indices=idx_cls)
            
            #    loss_cls = ∑(i∈cells) ∑(j∈anchors) ∑(c∈类别集合) U[i,j] * [p_[i,j,c] * -log(p[i,j,c]) + (1 - p_[i,j,c]) * -log(1 - p[i,j,c])]
            loss_cls = 1 * (- tf.math.log(cls_prob))
            loss_confidence = tf.math.reduce_mean(loss_cls)
            loss.append(loss_confidence)
            pass
        loss = tf.convert_to_tensor(loss)
        return loss
    pass



