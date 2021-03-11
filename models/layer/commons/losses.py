# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年3月10日
'''
import tensorflow as tf

import utils.alphabet as alphabet


#    YoloV4TinyLosses
class YoloV4BaseLosses(tf.keras.losses.Loss):

    def __init__(self,
                 **kwargs):
        super(YoloV4BaseLosses, self).__init__(**kwargs)
        pass
    
    #    loss_boxes
    def loss_boxes(self, liable_anchors, liable_num_objects, num_classes=len(alphabet.ALPHABET)):
        '''loss_box = ∑(i∈cell) ∑[j∈anchors] U[i,j] * (2 - Area(GT)) * CIoU(anchor[i,j], GT[i,j])
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
            @param liable_anchors: Tensor(sum_object, num_anchors, num_classes + 7 + 6)
                                            sum_object: 这批训练数据中负责预测的cell总数（与num_objects的划分对应）
                                            num_anchors: 每个cells有多少个anchor。配置决定，目前是3个
                                            num_classes + 7 + 6 + 1: 每个anchor信息
                                                num_classes: 各个分类得分
                                                7: anchor的[xl,yl, xr,yr], anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, 
                                                6: gt的[xl,yl, xr,yr, relative_area, idxV]
                                                1: 表示该位置anchor是否负责预测。0：不负责，1：负责
            @param liable_num_objects: Tensor(batch_size,)   每个batch实际含有的物体数，idxBHW第1个维度的划分
            @return: Tensor(batch_size, )
        '''
        #    取CIoU列    Tensor(sum_object, num_anchors, )
        ciou = liable_anchors[:, :, num_classes + 6]
        #    取相对面积列    Tensor(sum_object, num_anchors, )
        relative_area = liable_anchors[:, :, num_classes + 7 + 4]
        #    取掩码列    Tensor(sum_object, num_anchors, )
        mask = liable_anchors[:, :, num_classes + 13]
        
        #    计算 ∑(i∈cell) ∑[j∈anchors] U[i,j] * (2 - Area(GT)) * CIoU(anchor[i,j], GT[i,j])
        #    Tensor(sum_object, num_anchors, )
        loss = mask * (2 - relative_area) * ciou
        loss = tf.math.reduce_sum(loss, axis=-1)
        #    RaggedTensor(batch_size, num_objects)
        loss = tf.RaggedTensor.from_row_lengths(loss, row_lengths=liable_num_objects)
        loss = tf.math.reduce_mean(loss, axis=-1)
        return loss
    
    
    #    计算loss_confidence
    def loss_confidence(self, liable_anchors, liable_num_objects, num_classes=len(alphabet.ALPHABET)):
        '''loss_confidence负责计算负责预测的anchor的置信度损失（这部分与V3一致）：                            
            loss_confidence = ∑(i∈cells) ∑(j∈anchors) U[i,j] * [c_[i,j] * -log(c[i,j])]
                U[i,j] = 1 当第i个cell的第j个anchor负责物体时
                         0 当第i个cell的第j个anchor不负责物体时
                c_[i,j] = 第i个cell的第j个anchor负责物体的置信度
                            c_[i,j] = P(object) * IoU(anchor, box)
                                    = 1 * IoU(anchor, box)        （当anchor负责物体时，P(object)为1）
                c[i,j] = 第i个cell的第j个anchor预测负责物体的置信度
            @param liable_anchors: Tensor(sum_object, num_anchors, num_classes + 7 + 6)
                                            sum_object: 这批训练数据中负责预测的cell总数（与num_objects的划分对应）
                                            num_anchors: 每个cells有多少个anchor。配置决定，目前是3个
                                            num_classes + 7 + 6 + 1: 每个anchor信息
                                                num_classes: 各个分类得分
                                                7: anchor的[xl,yl, xr,yr], anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, 
                                                6: gt的[xl,yl, xr,yr, relative_area, idxV]
                                                1: 表示该位置anchor是否负责预测。0：不负责，1：负责
            @param liable_num_objects: Tensor(batch_size,)   每个batch实际含有的物体数，idxBHW第1个维度的划分
            @return: Tensor(batch_size, )
        '''
        #    取预测置信度    Tensor (sum_object, num_anchors, )
        confidence_prob = liable_anchors[:, :, num_classes + 4]
        #    取标记置信度    Tensor (sum_object, num_anchors, )
        confidence_true = liable_anchors[:, :, num_classes + 5]
        #    取掩码        Tensor (sum_object, num_anchors, )
        mask = liable_anchors[:, :, num_classes + 13]
        
        #    计算：loss_confidence = ∑(i∈cells) ∑(j∈anchors) U[i,j] * [c_[i,j] * -log(c[i,j])]
        #    Tensor (sum_object, num_anchors, )
        loss = mask * confidence_true * -tf.math.log(confidence_prob)
        loss = tf.math.reduce_sum(loss, axis=-1)
        loss = tf.RaggedTensor.from_row_lengths(loss, row_lengths=liable_num_objects)
        loss = tf.math.reduce_mean(loss, axis=-1)
        return loss
    
    
    #    计算loss_unconfidence
    def loss_unconfidence(self, unliable_anchors, unliable_sum_objects):
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
            @param unliable_anchors: Tensor(sum_unliable_cells, num_anchors, )
                                            sum_unliable: 本轮数据中所有不负责检测的cells总数
                                            num_anchors: 每个anchor的预测置信度
            @param unliable_sum_objects: Tensor(batch_size, )
                                             每个batch中不负责检测的cell总数
            @return: Tensor(batch_size, )
        '''
        #    计算loss_unconfidence = ∑(i∈cells) ∑(j∈anchors) Un[i,j] * [(1 - c_[i,j]) * -log(1 - c[i,j])]
        #    Tensor (sum_unliable_cells, num_anchors, )
        loss = -tf.math.log(unliable_anchors)
        loss = tf.math.reduce_sum(loss, axis=-1)
        loss = tf.RaggedTensor.from_row_lengths(loss, row_lengths=unliable_sum_objects)
        loss = tf.math.reduce_mean(loss, axis=-1)
        
        return loss
    
    
    #    计算loss_cls
    def loss_cls(self, liable_anchors, liable_num_objects, num_classes=len(alphabet.ALPHABET)):
        '''loss_cls负责计算负责预测anchor的分类损失：
            loss_cls负责计算负责预测anchor的分类损失：
                λ[cls] = 1
                loss_cls = ∑(i∈cells) ∑(j∈anchors) ∑(c∈类别集合) U[i,j] * [p_[i,j,c] * -log(p[i,j,c]) + (1 - p_[i,j,c]) * -log(1 - p[i,j,c])]
                U[i,j] = 1 当第i个cell的第j个anchor负责物体时
                         0 当第i个cell的第j个anchor不负责物体时
                p_[i,j,c] = 第i个cell中第j个anchor负责物体的实际从属分类概率（one_hot）
                p[i,j,c] = 第i个cell中第j个anchor预测物体属于第c类的概率
            @param liable_anchors: Tensor(sum_object, num_anchors, num_classes + 7 + 6)
                                            sum_object: 这批训练数据中负责预测的cell总数（与num_objects的划分对应）
                                            num_anchors: 每个cells有多少个anchor。配置决定，目前是3个
                                            num_classes + 7 + 6 + 1: 每个anchor信息
                                                num_classes: 各个分类得分
                                                7: anchor的[xl,yl, xr,yr], anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, 
                                                6: gt的[xl,yl, xr,yr, relative_area, idxV]
                                                1: 表示该位置anchor是否负责预测。0：不负责，1：负责
            @param liable_num_objects: Tensor(batch_size,)   每个batch实际含有的物体数，idxBHW第1个维度的划分
            @return: Tensor(batch_size, )
        '''
        #    取各个分类得分                 Tensor (sum_object, num_anchors, num_classes)
        cls_prob = liable_anchors[:, :, :num_classes]
        #    取标记分类索引，并做成one_hot    Tensor (sum_object, num_anchors, num_classes)
        cls_true = tf.cast(liable_anchors[:, :, num_classes + 7 + 5], dtype=tf.int32)
        cls_true = tf.one_hot(cls_true, depth=num_classes)
        #    取掩码    Tensor (sum_object, num_anchors, 1)
        mask = liable_anchors[:, :, num_classes + 7 + 6]
        mask = tf.expand_dims(mask, axis=-1)
        
        #    计算: loss_cls = ∑(i∈cells) ∑(j∈anchors) ∑(c∈类别集合) U[i,j] * [p_[i,j,c] * -log(p[i,j,c]) + (1 - p_[i,j,c]) * -log(1 - p[i,j,c])]
        #    Tensor (sum_object, num_anchors, num_classes)
        loss = mask * (cls_true * -tf.math.log(cls_prob) + (1 - cls_true) * -tf.math.log(1 - cls_prob))
        loss = tf.math.reduce_sum(loss, axis=(1, 2))
        loss = tf.RaggedTensor.from_row_lengths(loss, row_lengths=liable_num_objects)
        loss = tf.math.reduce_mean(loss, axis=-1)
        return loss
    

    pass

