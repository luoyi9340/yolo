# -*- coding: utf-8 -*-  
'''
IoU相关

@author: luoyi
Created on 2021年2月23日
'''
import numpy as np
import tensorflow as tf
import math


#    常量定义
#    (2/π)²
V_PI = (2. / math.pi) ** 2


#    批量计算多个box(srcs)与1个目标box(tag)的IoU
def iou_b_n21_tf(rect_srcs, rect_tag):
    '''
        @param rect_srcs: tensor(batch, num, 4)
        @param rect_tag: tensor(batch, 4)
        @return tensor(num_object, num)
    '''
    #    将rect_tag扩展为tensor(batch, 1, 4)，并复制为tensor(batch, num, 4)
    rect_tag = tf.expand_dims(rect_tag, axis=1)
    rect_tag = tf.repeat(rect_tag, repeats=rect_srcs.shape[1], axis=1)
    
    #    取交点处坐标 tensor(batch, num)
    xl = tf.math.maximum(rect_srcs[:,:,0], rect_tag[:,:,0])
    yl = tf.math.maximum(rect_srcs[:,:,1], rect_tag[:,:,1])
    xr = tf.math.minimum(rect_srcs[:,:,2], rect_tag[:,:,2])
    yr = tf.math.minimum(rect_srcs[:,:,3], rect_tag[:,:,3])
    #    取交点区域长宽 tensor(batch, num)
    w = tf.math.maximum(0., xr - xl)
    h = tf.math.maximum(0., yr - yl)
    #    计算tag, srcs面积 tensor(batch, num)
    area_tag = tf.math.abs(rect_tag[:,:,3] - rect_tag[:,:,1]) * tf.math.abs(rect_tag[:,:,2] - rect_tag[:,:,0])
    area_srcs = tf.math.abs(rect_srcs[:,:,3] - rect_srcs[:,:,1]) * tf.math.abs(rect_srcs[:,:,2] - rect_srcs[:,:,0])
    #    交集区域面积 tensor(batch, num)
    areas_intersection = w * h
    #    计算IoU tensor(batch, num)，并转换为tensor(batch, num, 1)
    iou = areas_intersection / ((area_tag + area_srcs) - areas_intersection)
    return iou

#    批量计算多个box(srcs)与1个目标box(tag)的CIoU
def ciou_b_n21_tf(rect_srcs, rect_tag):
    '''ciou = 1 - IoU(A, B) + d²(A[cx,cy] - B[cx,cy])/c² + α * v
            其中：α * v为宽高比的惩罚项
                    d为欧氏距离公式
                    A[cx,cy]、B[cx,cy]为A，B两个预测框的中心点坐标
                    c为包围A，B最小矩形的对角线长度
                    v = (2/π)² * [arctan(A[w]/B[h]) - arctan(B[w]/B[h])]²
                    α = v / [(i - IoU) + v]
                    当A与B的宽高很接近，惩罚项就接近于0
        @param rect_srcs: tensor(batch, num, 4)
        @param rect_tag: tensor(batch, 4)
        @return tensor(batch, num)
    '''
    #    计算IoU
    iou = iou_b_n21_tf(rect_srcs, rect_tag)
    
    #    将rect_tag扩展为tensor(batch, 1, 4)，并复制为tensor(batch, num, 4)
    rect_tag = tf.expand_dims(rect_tag, axis=1)
    rect_tag = tf.repeat(rect_tag, repeats=rect_srcs.shape[1], axis=1)
    
    #    计算d²(A[cx,cy] - B[cx,cy])/c²
    Acx = (rect_srcs[:,:,0] + rect_srcs[:,:,2]) / 2             #    tensor(batch, num)
    Acy = (rect_srcs[:,:,1] + rect_srcs[:,:,3]) / 2             #    tensor(batch, num)
    Bcx = (rect_tag[:,:,0] + rect_tag[:,:,2]) / 2               #    tensor(batch, num)
    Bcy = (rect_tag[:,:,1] + rect_tag[:,:,3]) / 2               #    tensor(batch, num)
    d_square = tf.math.square(Bcx - Acx) + tf.math.square(Bcy - Acy)        #    tensor(batch, num)
    surround_box_xl = tf.math.minimum(rect_srcs[:,:,0], rect_tag[:,:,0])    #    tensor(batch, num)
    surround_box_yl = tf.math.minimum(rect_srcs[:,:,1], rect_tag[:,:,1])    #    tensor(batch, num)
    surround_box_xr = tf.math.maximum(rect_srcs[:,:,2], rect_tag[:,:,2])    #    tensor(batch, num)
    surround_box_yr = tf.math.maximum(rect_srcs[:,:,3], rect_tag[:,:,3])    #    tensor(batch, num)
    c_square = tf.math.square(surround_box_xr - surround_box_xl) + tf.math.square(surround_box_yr - surround_box_yl)    #    tensor(batch, num)
    p1 = d_square / c_square
    
    #    计算α * v
    #    v = (2/π)² * [arctan(A[w]/A[h]) - arctan(B[w]/B[h])]²
    #    α = v / [(i - IoU) + v]
    Aw = tf.math.abs(rect_srcs[:,:,2] - rect_srcs[:,:,0])
    Ah = tf.math.abs(rect_srcs[:,:,3] - rect_srcs[:,:,1])
    Bw = tf.math.abs(rect_tag[:,:,2] - rect_tag[:,:,0])
    Bh = tf.math.abs(rect_tag[:,:,3] - rect_tag[:,:,1])
    v = V_PI  * tf.math.square(tf.math.atan(Aw / Ah) - tf.math.atan(Bw / Bh))
    alpha = v / ((1 - iou) + v)
    
    #    CIoU = 1 - IoU(A, B) + d²(A[cx,cy] - B[cx,cy])/c² + α * v
    ciou = 1 - iou + p1 + alpha * v
    return ciou


#    计算多个的box（srcs）与一个目标box（tag）的IoU
def iou_n21_np(rect_srcs, rect_tag):
    '''
        @param rect_srcs: 多个box numpy(num, 4) [xl,yl, xr,yr]
        @param tect_tag: 单个box numpy(4,) [xl,yl, xr,yr]
        @return: IoU numpy(num, 1)
    '''
    #    取交点处坐标
    xl = np.maximum(rect_tag[0], rect_srcs[:,0], dtype=np.float32)
    yl = np.maximum(rect_tag[1], rect_srcs[:,1], dtype=np.float32)
    xr = np.minimum(rect_tag[2], rect_srcs[:,2], dtype=np.float32)
    yr = np.minimum(rect_tag[3], rect_srcs[:,3], dtype=np.float32)
    #    取交点区域长宽
    w = np.maximum(0., xr - xl)
    h = np.maximum(0., yr - yl)
    #    计算tag, srcs面积
    area_tag = abs(rect_tag[3] - rect_tag[1]) * abs(rect_tag[2] - rect_tag[0])
    area_srcs = np.abs(rect_srcs[:,3] - rect_srcs[:,1]) * np.abs(rect_srcs[:,2] - rect_srcs[:,0])
    areas_intersection = w * h
    iou = areas_intersection / ((area_tag + area_srcs) - areas_intersection)
    #    如果iou > 1则将其置为1
    iou[iou > 1] = 1
    return iou


#    计算两个box的IoU
def iou_121_np(box1, box2):
    '''
        @param box1: 物体1 [xl,yl, xr,yr]
        @param box2: 物体2 [xl,yl, xr,yr]
    '''
        #    取交点处坐标
    xl = np.maximum(box1[0], box2[0], dtype=np.float32)
    yl = np.maximum(box1[1], box2[1], dtype=np.float32)
    xr = np.minimum(box1[2], box2[2], dtype=np.float32)
    yr = np.minimum(box1[3], box2[3], dtype=np.float32)
    #    取交点区域长宽
    w = np.maximum(0., xr - xl)
    h = np.maximum(0., yr - yl)
    #    计算tag, srcs面积
    area_tag = abs(box1[3] - box1[1]) * abs(box1[2] - box1[0])
    area_srcs = np.abs(box2[3] - box2[1]) * np.abs(box2[2] - box1[0])
    areas_intersection = w * h
    iou = areas_intersection / ((area_tag + area_srcs) - areas_intersection)
    #    如果iou > 1则将其置为1
    if (iou > 1): iou = 1
    return iou



