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
import utils.conf as conf
import utils.alphabet as alphabet
from models.layer.commons.losses import YoloV4BaseLosses
from models.layer.commons.part import YoloHardRegister, AnchorsRegister
from models.layer.commons.preporcess import takeout_liables, takeout_unliables, parse_idxBHW


#    YOLO V4的loss
class YoloLosses(YoloV4BaseLosses):
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
        '''
            @param yolohard_register: yolohard结果暂存
            @param anchors_register: yolohard解析为anchors结果暂存
        '''
        super(YoloLosses, self).__init__(**kwargs)
        
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
    
    #    计算loss
    def call(self, y_true, y_pred):
        '''
            @param y_true: tensor(batch_size, num_scale, 6, 2 + 5 + num_anchor * 3)
            @param y_pred: 用不到，x数据从yolohard_register中拿
        '''
        #    解析yolohard值并暂存
        yolohard1 = self._yolohard_register.get_yolohard1()
        y_true1 = y_true[:, 2, :]
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
        
        yolohard3 = self._yolohard_register.get_yolohard3()
        y_true3 = y_true[:, 0, :]
        liable_idxBHW3, liable_num_objects3 = parse_idxBHW(y_true3, self._batch_size)
        liable_anchors3, liable_num_objects3 = takeout_liables(liable_idxBHW3, liable_num_objects3, yolohard3, y_true3, self._num_anchors, self._batch_size, self._num_classes, self._threshold_liable_iou)
        unliable_anchors3, unliable_num_objects3 = takeout_unliables(liable_idxBHW3, liable_num_objects3, yolohard3, y_true3, self._batch_size, self._num_anchors, self._num_classes)
        self._anchors_register.deposit_yolohard3(liable_anchors2, liable_num_objects3, unliable_anchors3, unliable_num_objects3)
        
        
        #    计算loss
        loss_boxes1 = self.loss_boxes(liable_anchors1, liable_num_objects1, self._num_classes)
        loss_confidence1 = self.loss_confidence(liable_anchors1, liable_num_objects1, self._num_classes)
        loss_unconfidence1 = self.loss_unconfidence(unliable_anchors1, unliable_num_objects1)
        loss_cls1 = self.loss_cls(liable_anchors1, liable_num_objects1, self._num_classes)
        
        loss_boxes2 = self.loss_boxes(liable_anchors2, liable_num_objects2, self._num_classes)
        loss_confidence2 = self.loss_confidence(liable_anchors2, liable_num_objects2, self._num_classes)
        loss_unconfidence2 = self.loss_unconfidence(unliable_anchors2, unliable_num_objects2)
        loss_cls2 = self.loss_cls(liable_anchors2, liable_num_objects2, self._num_classes)
        
        loss_boxes3 = self.loss_boxes(liable_anchors3, liable_num_objects3, self._num_classes)
        loss_confidence3 = self.loss_confidence(liable_anchors3, liable_num_objects3, self._num_classes)
        loss_unconfidence3 = self.loss_unconfidence(unliable_anchors3, unliable_num_objects3)
        loss_cls3 = self.loss_cls(liable_anchors3, liable_num_objects3, self._num_classes)
        
        
        loss_boxes = (loss_boxes1 + loss_boxes2 + loss_boxes3) / 3
        loss_confidence = (loss_confidence1 + loss_confidence2 + loss_confidence3) / 3
        loss_unconfidence = (loss_unconfidence1 + loss_unconfidence2 + loss_unconfidence3) / 3
        loss_cls = (loss_cls1 + loss_cls2 + loss_cls3) / 3
        loss = self._loss_lamda_box * loss_boxes \
                + self._loss_lamda_confidence * loss_confidence \
                + self._loss_lamda_unconfidence * loss_unconfidence \
                + self._loss_lamda_cls * loss_cls
                
        return loss

    pass



