# -*- coding: utf-8 -*-  
'''
非极大值抑制

步骤：
step1: 置信度降序对应的anchor_box取第1个，记录进最终判定
step2: 计算第1个与其他box的IoU，超过阈值的判定为重复。从生下的anchor_box中过滤掉重复的box
step3: 在生下的anchor_box中取第1个，记录进最终判定，重复上述步骤直到没有anchor_box

@author: luoyi
Created on 2021年3月13日
'''
import tensorflow as tf

import utils.conf as conf
from utils.iou import iou_n21_tf


#    非极大值抑制
def nms_tf(anchors_boxes, threshold_overlap_iou=conf.V4.get_threshold_overlap_iou()):
    '''
        @param anchors_boxes: Tensor(sum_qualified_anchors, 4)
        @return: Tensor(num_anchors, 4)
    '''
    res = []
    #    取第1个anchor，保存并删除
    anchor = anchors_boxes[0]               #    Tensor(4,)
    res.append(anchor)
    anchors_boxes = anchors_boxes[1:, :]
    
    while (anchors_boxes.shape[0] > 0):
        #    计算anchor与每个剩下的anchor的IoU
        iou = iou_n21_tf(rect_srcs=anchors_boxes, rect_tag=anchor)
        #    取小于阈值的IoU的位置，作为保留anchors_boxes
        anchors_boxes = tf.gather_nd(anchors_boxes, indices=tf.where(iou < threshold_overlap_iou))
        
        if (anchors_boxes is None or anchors_boxes.shape[0] == 0): break
        
        #    取第1个anchor保存进res，循环执行此逻辑
        anchor = anchors_boxes[0]
        res.append(anchor)
        anchors_boxes = anchors_boxes[1:, :]
        pass
    
    return tf.stack(res, axis=0)
