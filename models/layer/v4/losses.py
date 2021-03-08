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
    def loss_box(self, liable_anchors, num_classes=len(alphabet.ALPHABET)):
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
                            
            @param liable_anchors RaggedTensor(batch_size, num_object, num_liable, num_classes + 7 + 6)
                                            batch_size: 图片批量个数
                                            num_object: 实际物体数，每张图片可能不一样
                                            num_liable: 负责预测的anchor数，每个物体可能不一样(0,3]
                                            num_classes + 7 + 6: 分类数 + anchor信息 + gt信息
                                                num_classes: 各个分类得分
                                                7: anchor的[xl,yl, xr,yr], anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, 
                                                6: gt的[xl,yl, xr,yr, relative_area, idxV]
            @return: Tensor(batch_size, 1)
        '''
        #    取gt相对面积 RaggedTensor(batch_size, num_object(不固定), num_liable(不固定), 1)
        relative_area = liable_anchors[:, :, :, num_classes + 7 + 4 : num_classes + 7 + 4 + 1]
        #    取ciou RaggedTensor(batch_size, num_object(不固定), num_liable(不固定), 1)
        ciou = liable_anchors[:, :, :, num_classes + 6 : num_classes + 6 + 1]
        
        #    计算∑(i∈cell) ∑[j∈anchors] U[i,j] * (2 - Area(GT)) * CIoU(anchor[i,j], GT[i,j]) 
        #    RaggedTensor(batch_size, num_object(不固定), num_liable(不固定), 1)
        loss = (2 - relative_area) * ciou
        loss = tf.math.reduce_mean(loss, axis=(1,2,3))
        return loss
    
    #    计算loss_confidence
    def loss_confidence(self, liable_anchors, num_classes=len(alphabet.ALPHABET)):
        '''loss_confidence负责计算负责预测的anchor的置信度损失（这部分与V3一致）：                            
            loss_confidence = ∑(i∈cells) ∑(j∈anchors) U[i,j] * [c_[i,j] * -log(c[i,j])]
                U[i,j] = 1 当第i个cell的第j个anchor负责物体时
                         0 当第i个cell的第j个anchor不负责物体时
                c_[i,j] = 第i个cell的第j个anchor负责物体的置信度
                            c_[i,j] = P(object) * IoU(anchor, box)
                                    = 1 * IoU(anchor, box)        （当anchor负责物体时，P(object)为1）
                c[i,j] = 第i个cell的第j个anchor预测负责物体的置信度
            @params liable_anchors RaggedTensor(batch_size, num_object, num_liable, num_classes + 7 + 6)
                                            batch_size: 图片批量个数
                                            num_object: 实际物体数，每张图片可能不一样
                                            num_liable: 负责预测的anchor数，每个物体可能不一样(0,3]
                                            num_classes + 7 + 6: 分类数 + anchor信息 + gt信息
                                                num_classes: 各个分类得分
                                                7: anchor的[xl,yl, xr,yr], anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, 
                                                6: gt的[xl,yl, xr,yr, relative_area, idxV]
            @return: Tensor(batch_size, 1)
        '''
        #    取预测置信度    RaggedTensor(batch_size, num_object(不固定), num_liable(不固定), 1)
        confidence_prob = liable_anchors[:, :, :, num_classes + 4 : num_classes + 4 + 1]
        #    取标记置信度    RaggedTensor(batch_size, num_object(不固定), num_liable(不固定), 1)
        confidence_true = liable_anchors[:, :, :, num_classes + 5 : num_classes + 5 + 1]
        
        #    计算: ∑(i∈cells) ∑(j∈anchors) U[i,j] * [c_[i,j] * -log(c[i,j])]
        #    RaggedTensor(batch_size, num_object(不固定), num_liable(不固定), 1)
        loss = confidence_true * -tf.math.log(confidence_prob)
        loss = tf.math.reduce_mean(loss, axis=(1, 2, 3))
        return loss
    
    #    计算loss_unconfidence
    def loss_unconfidence(self, unliable_anchors):
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
            @params unliable_anchors: Tensor(batch_size, H, W, num_anchors)
            @return: Tensor(batch_size, 1)
        '''
        #    计算: ∑(i∈cells) ∑(j∈anchors) Un[i,j] * [(1 - c_[i,j]) * -log(1 - c[i,j])]
        #    Tensor(batch_size, H, W, num_anchors)
        loss = 1 * -tf.math.log(1 - unliable_anchors)
        loss = tf.math.reduce_mean(loss, axis=(1, 2, 3))
        return loss
    
    #    计算loss_cls
    def loss_cls(self, liable_anchors, num_classes=len(alphabet.ALPHABET)):
        '''loss_cls负责计算负责预测anchor的分类损失：
            loss_cls负责计算负责预测anchor的分类损失：
                λ[cls] = 1
                loss_cls = ∑(i∈cells) ∑(j∈anchors) ∑(c∈类别集合) U[i,j] * [p_[i,j,c] * -log(p[i,j,c]) + (1 - p_[i,j,c]) * -log(1 - p[i,j,c])]
                U[i,j] = 1 当第i个cell的第j个anchor负责物体时
                         0 当第i个cell的第j个anchor不负责物体时
                p_[i,j,c] = 第i个cell中第j个anchor负责物体的实际从属分类概率（one_hot）
                p[i,j,c] = 第i个cell中第j个anchor预测物体属于第c类的概率
            @params liable_anchors RaggedTensor(batch_size, num_object, num_liable, num_classes + 7 + 6)
                                            batch_size: 图片批量个数
                                            num_object: 实际物体数，每张图片可能不一样
                                            num_liable: 负责预测的anchor数，每个物体可能不一样(0,3]
                                            num_classes + 7 + 6: 分类数 + anchor信息 + gt信息
                                                num_classes: 各个分类得分
                                                7: anchor的[xl,yl, xr,yr], anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, 
                                                6: gt的[xl,yl, xr,yr, relative_area, idxV]
            @return: Tensor(batch_size, 1)
        '''
        #    取预测的分类得分    RaggedTensor(batch_size, num_object, num_liable, num_classes)
        cls_prob = liable_anchors[:, :, :, :num_classes]
        cls_prob = tf.expand_dims(cls_prob, axis=-2)        #    下面one_hot会增加1个维度，这里先补对齐
        #    取标记的分类得分    RaggedTensor(batch_size, num_object, num_liable, num_classes)
        cls_true = liable_anchors[:, :, :, num_classes + 7 + 5 : num_classes + 7 + 6]
        cls_true = tf.cast(cls_true, dtype=tf.int32)
        cls_true = tf.one_hot(cls_true, depth=num_classes)
        
        #    计算: ∑(i∈cells) ∑(j∈anchors) ∑(c∈类别集合) U[i,j] * [p_[i,j,c] * -log(p[i,j,c]) + (1 - p_[i,j,c]) * -log(1 - p[i,j,c])]
        #    RaggedTensor(batch_size, num_object, num_liable, num_classes)
        loss = cls_true * -tf.math.log(cls_prob) + (1 - cls_true) * -tf.math.log(1 - cls_prob)
        #    RaggedTensor(batch_size, num_object, num_liable)
        loss = tf.math.reduce_sum(loss, axis=-1)
        loss = tf.math.reduce_mean(loss, axis=(1, 2))
        loss = tf.squeeze(loss)
        return loss
    pass



