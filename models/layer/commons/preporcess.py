# -*- coding: utf-8 -*-  
'''
yolo v4 tiny 预处理


数据维度描述：
batch_size: 1个batch_size代表1张图片

num_objects: 每张图的物体数量，每张图不相等
                1个物体对应1个cell
    
num_anchors: 每个cell中的anchor个数。
                配置统一决定。这里是3
                
num_liable_anchors: 每个cell负责预测的anchor数量
                        - IoU超过阈值的anchor
                        - IoU最大的anchor，当所有IoU都小于阈值时
                        



@author: luoyi
Created on 2021年3月9日
'''
import tensorflow as tf

import utils.conf as conf
import utils.alphabet as alphabet
from utils.iou import iou_b_n2n_tf, ciou_b_n2n_tf

#    从yolohard中取负责检测的部分
def takeout_liables(liable_idxBHW=None,
                    liable_num_objects=None,
                    yolohard=None, 
                    y_true=None, 
                    num_anchors=conf.DATASET_CELLS.get_anchors_set().shape[1],
                    batch_size=conf.DATASET_CELLS.get_batch_size(),
                    num_classes=len(alphabet.ALPHABET),
                    threshold_liable_iou=conf.V4.get_threshold_liable_iou()):
    '''从yolohard中取负责检测的和不负责检测的cells
        @param yolohard: 模型产出的预测 Tensor(batch_size, H, W, num_anchors, num_classes + 5)
                                                        num_classes: 各个分类得分
                                                        1：预测置信度
                                                        4：预测的dx, dy, dw, dh
        @param y_true: 与yolohard对应的y_true Tensor(batch_size, 6, 2 + 5 + num_anchor * 3)
                                                        2: idxH, idxW (-1,-1)代表是填充的值
                                                        5: x,y,w,h, idxV
                                                        num_anchor*3: IoU, w, h
        @param num_anchors: 每个cell中anchor数量
        @param batch_size: 批量大小
        @return: liable_anchors:    Tensor (sum_object, num_anchors, num_classes + 7 + 6)
                                            sum_object: 这批训练数据中负责预测的cell总数（与num_objects的划分对应）
                                            num_anchors: 每个cells有多少个anchor。配置决定，目前是3个
                                            num_classes + 7 + 6 + 1: 每个anchor信息
                                                num_classes: 各个分类得分
                                                7: anchor的[xl,yl, xr,yr], anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, 
                                                6: gt的[xl,yl, xr,yr, relative_area, idxV]
                                                1: 表示该位置anchor是否负责预测。0：不负责，1：负责
                liable_num_objects: Tensor(batch_size,)   每个batch实际含有的物体数，idxBHW第1个维度的划分
    '''
    if (liable_idxBHW is None or liable_num_objects is None): liable_idxBHW, liable_num_objects = parse_idxBHW(y_true, batch_size)
    sum_object = tf.math.reduce_sum(liable_num_objects)
    fmaps_shape = (yolohard.shape[1], yolohard.shape[2])
    
    #    yolohard中负责预测的cell预测信息        Tensor (sum_object, num_anchors, num_classes + 5)
    liable_anchors_prob = tf.gather_nd(yolohard, indices=liable_idxBHW)
    #    y_true中负责预测的cell标记信息
    liable_y = y_true[y_true[:, :, 0] >= 0]                                                  #    Tensor(sum_object, 2 + 5 + num_anchor * 3)
    #    取cell左上点坐标，并扩展为 Tensor(sum_object, 1, 2)用于计算时的广播
    liable_y_cell_idx = liable_y[:, :2]
    liable_y_cell_idx = tf.expand_dims(liable_y_cell_idx, axis=-2)
    #    取anchors信息，并整形为 Tensor(sum_object, num_anchors, 3)
    liable_y_anchors_info = liable_y[:, 7:]
    liable_y_anchors_info = tf.reshape(liable_y_anchors_info, shape=(sum_object, num_anchors, 3))

    #    还原gts_boxes    Tensor(sum_object, num_anchors, 6)
    gts_boxes = reduction_gts_boxes(liable_y, fmaps_shape, num_anchors)
    #    还原anchors_boxes    Tensor(sum_object, num_anchors, 4)
    anchors_boxes = reduction_anchors_boxes(liable_anchors_prob, liable_y_cell_idx, liable_y_anchors_info, fmaps_shape, num_classes)
    
    #    计算anchors与gt的IoU    Tensor(sum_object, num_anchors, )
    iou = iou_b_n2n_tf(rect_srcs=anchors_boxes, rect_tags=gts_boxes)
    
    #    需要的返回值信息
    #    取各个anchors的分类得分    Tensor(sum_object, num_anchors, num_classes)
    liable_anchors_cls = liable_anchors_prob[:, :, :num_classes]
    #    anchors_boxes信息    Tensor(sum_object num_anchors, 4)
    liable_anchors_boxes = anchors_boxes
    #    anchors置信度预测    Tensor(sum_object num_anchors, )
    liable_anchors_confidence_prob = liable_anchors_prob[:, :, num_classes]
    #    anchors置信度标记    Tensor(sum_object num_anchors, )
    liable_anchors_confidence_true = liable_y_anchors_info[:, :, 0]
    #    计算anchors与gt的CIoU    Tensor(sum_object, num_anchors, ) 注：存在冗余计算，但相比于挑出有效anchors的计算代价，多了就多了
    liable_anchors_ciou = ciou_b_n2n_tf(rect_srcs=anchors_boxes, rect_tags=gts_boxes, iou=iou)
    #    gts_boxes信息    Tensor(sum_object, num_anchors, 6)
    liable_anchors_gts = gts_boxes
    #    anchors是否负责预测的掩码。0：不负责，1：负责    Tensor(sum_object, num_anchors,)
    liable_anchors_mask = liable_mask(iou, threshold_liable_iou)
    #    组合需要的返回值信息
    liable_anchors = tf.concat([liable_anchors_cls,
                                liable_anchors_boxes,
                                tf.expand_dims(liable_anchors_confidence_prob, axis=-1),
                                tf.expand_dims(liable_anchors_confidence_true, axis=-1),
                                tf.expand_dims(liable_anchors_ciou, axis=-1),
                                liable_anchors_gts,
                                tf.expand_dims(liable_anchors_mask, axis=-1)], axis=-1)
    return liable_anchors, liable_num_objects


#    取负责预测的cell的idxH, idxW，和每个batch中的实际物体数
def parse_idxBHW(y_true, batch_size=conf.DATASET_CELLS.get_batch_size()):
    '''
        @param y_true: 与yolohard对应的y_true Tensor(batch_size, 6, 2 + 5 + num_anchor * 3)
                                                        2: idxH, idxW (-1,-1)代表是填充的值
                                                        5: x,y,w,h, idxV
                                                        num_anchor*3: IoU, w, h
        @return: idxBHW Tensor(负责预测的cell总数, 3)
                 num_object Tensor(batch_size,)   每个batch实际含有的物体数，idxBHW第1个维度的划分
    '''
    #    取负责预测的cells的索引，(-1,-1)说明是填充数据  Tensor(batch_size, 6, 2)
    idxHW = tf.cast(y_true[:, :, :2], dtype=tf.int64)
    #    求每个batch实际有效物体数    Tensor(batch_size, )
    num_objects = tf.math.count_nonzero(idxHW[:, :, 0] >= 0, axis=1)

    #    idxHW扩展为idxBHW
    idxB = tf.range(batch_size, dtype=tf.int64)
    idxB = tf.expand_dims(idxB, axis=-1)
    idxB = tf.repeat(idxB, repeats=idxHW.shape[1], axis=0)
    idxHW = tf.reshape(idxHW, shape=(batch_size * idxHW.shape[1], idxHW.shape[2]))
    idxBHW = tf.concat([idxB, idxHW], axis=-1)
    #    负责预测物体的cell的idx      Tensor(负责预测物体数, 2)
    idxBHW = idxBHW[idxBHW[:, 1] >= 0]
    
    return idxBHW, num_objects


#    liable_y中的信息还原为gts_boxes
def reduction_gts_boxes(liable_y, fmaps_shape, num_anchors):
    '''
        @param liable_y: Tensor(sum_object, 2 + 5 + num_anchor * 3)
                                2: cell的(idxH, idxW)
                                5: x,y(相对左上点的偏移),w,h(整图占比), idxV
        @param fmaps_shape: 特征图高宽(H, W)
        @param num_anchors: 每个cell的anchor数量
        @return gtx_boxes Tensor(sum_object, num_anchor, 4 + 1 + 1)
                                    4: 相对特征图的lx,ly, rx,ry
                                    1: 相对面积（宽高占比乘出来的）
                                    1: 分类索引
    '''
    #    cell左上点坐标
    cell_idx = liable_y[:, :2]
    #    gt的x,y, w,h
    gts_loc = liable_y[:, 2:6]
    
    #    相对面积    Tensor (sum_object, )
    relative_area = gts_loc[:, 2] * gts_loc[:, 3]
    #    分类索引    Tensor (sum_object, )
    idxV = liable_y[:, 6]
    
    #    还原坐标信息
    gtx_cx = gts_loc[:, 0] + cell_idx[:, 1]
    gtx_cy = gts_loc[:, 1] + cell_idx[:, 0]
    gtx_half_w = gts_loc[:, 2] * fmaps_shape[1] / 2
    gtx_half_h = gts_loc[:, 3] * fmaps_shape[0] / 2
    gtx_lx = gtx_cx - gtx_half_w                #    Tensor (sum_object, )
    gtx_ly = gtx_cy - gtx_half_h                #    Tensor (sum_object, )
    gtx_rx = gtx_cx + gtx_half_w                #    Tensor (sum_object, )
    gtx_ry = gtx_cy + gtx_half_h                #    Tensor (sum_object, )
    
    #    整合信息 gts_boxes Tensor (sum_object, 6)
    gts_boxes = tf.stack([gtx_lx, gtx_ly, gtx_rx, gtx_ry, 
                          relative_area,
                          idxV], axis=-1)
    #    扩展为Tensor (sum_object, num_anchors, 6)
    gts_boxes = tf.repeat(tf.expand_dims(gts_boxes, axis=-2), repeats=num_anchors, axis=-2)
    return gts_boxes


#    liable_anchor_prob中的信息还原为anchors_boxes
def reduction_anchors_boxes(liable_anchors_prob, liable_y_cell_idx, liable_y_anchors_info, fmaps_shape, num_classes):
    '''liable_anchors_prob中的预测dx,dy,dw,dh还原为anchors_boxes
        @param liable_anchors_prob: Tensor (sum_object, num_anchors, num_classes + 5)
                                                num_classes: 分类数量
                                                1: 预测置信度
                                                4: dx,dy, dw,dh
        @param liable_y_cell_idx: cell左上点坐标，Tensor(sum_object, 1, 2)
        @param liable_y_anchors_info: anchors信息，Tensor(sum_object, num_anchors, 3)
        @param fmaps_shape: 特征图尺寸(H, W)
        @return: anchors_boxes Tensor (sum_object, num_anchors, 4)
                                        4: lx,ly, rx,ry (相对特征图坐标)
    '''
    #    取dx,dy,dw,dh
    dn = liable_anchors_prob[:, :, num_classes + 1:]
    #    还原中心点坐标, 宽高信息
    cx = dn[:, :, 0] + liable_y_cell_idx[:, :, 1]                                                    #    cx = dx + cell[x]
    cy = dn[:, :, 1] + liable_y_cell_idx[:, :, 0]                                                    #    cy = dy + cell[y]
    half_w = tf.math.exp(dn[:, :, 2]) *  (liable_y_anchors_info[:, :, 1] * fmaps_shape[1]) / 2       #    w = exp(dw) * anchor[w]
    half_h = tf.math.exp(dn[:, :, 3]) *  (liable_y_anchors_info[:, :, 2] * fmaps_shape[0]) / 2       #    h = exp(dh) * anchor[h]
    anchor_lx = cx - half_w
    anchor_ly = cy - half_h
    anchor_rx = cx + half_w
    anchor_ry = cy + half_h
    
    #    组合成anchors_boxes    Tensor(sum_object, num_anchors, 4)
    anchors_boxes = tf.stack([anchor_lx, anchor_ly, anchor_rx, anchor_ry], axis=-1)
    return anchors_boxes


#    根据IoU和阈值生成anchor是否负责预测的掩码
def liable_mask(iou, threshold_liable_iou=conf.V4.get_threshold_liable_iou()):
    ''' IoU超过阈值的位置判定为负责预测的anchor
            如果1个cell内的所有anchor的IoU都小于阈值，则取最大IoU
        @param iou: anchors_boxes与gts_boxes的IoU Tensor(sum_object, num_anchors, )
        @param threshold_liable_iou: 负责预测的anchor与gt的IoU阈值。超过阈值的anchor都会被判定为负责预测的anchor
        @return Tensor(sum_object, num_anchors, )
    '''
    #    取每个anchors的最大IoU    Tensor(sum_object, 1)
    max_iou = tf.math.reduce_max(iou, axis=-1)
    max_iou = tf.expand_dims(max_iou, axis=-1)
    
    iou_one_tmp = tf.ones_like(iou)
    iou_zero_tmp = tf.zeros_like(iou)
    
    #    取小于阈值但是等于最大值的IoU，让他+1。这样小于阈值的最大IoU就会超过阈值
    condition = tf.logical_and(iou < threshold_liable_iou, tf.equal(iou, max_iou))
    iou_add = tf.where(condition, iou_one_tmp, iou_zero_tmp)
    iou = iou + iou_add
    
    #    取IoU ≥ 阈值的位置置为1，其他置为0    Tensor(sum_object, num_anchors,)
    mask = tf.where(iou >= threshold_liable_iou, iou_one_tmp, iou_zero_tmp)
    return mask


#    取不负责预测的anchors信息
def takeout_unliables(liable_idxBHW=None, 
                      liable_num_objects=None, 
                      yolohard=None, 
                      y_true=None, 
                      batch_size=conf.DATASET_CELLS.get_batch_size(),
                      num_anchors=conf.DATASET_CELLS.get_anchors_set().shape[1],
                      num_classes=len(alphabet.ALPHABET)):
    '''取不负责预测的anchor信息
        @param idxBHW: 取负责预测的cell索引 Tensor(负责预测的cell总数, 3)
        @param num_objects: 每个batch负责预测的cell数 Tensor(batch_size, )
        @param yolohard: 模型产出的预测 Tensor(batch_size, H, W, num_anchors, num_classes + 5)
                                                        num_classes: 各个分类得分
                                                        1：预测置信度
                                                        4：预测的dx, dy, dw, dh
        @param y_true: 与yolohard对应的y_true Tensor(batch_size, 6, 2 + 5 + num_anchor * 3)
                                                        2: idxH, idxW (-1,-1)代表是填充的值
                                                        5: x,y,w,h, idxV
                                                        num_anchor*3: IoU, w, h
        @return: unliable_anchors: Tensor(sum_unliable_cells, num_anchors, )
                                            sum_unliable: 本轮数据中所有不负责检测的cells总数
                                            num_anchors: 每个anchor的预测置信度
                 unliable_sum_objects: Tensor(batch_size, )
                                             每个batch中不负责检测的cell总数
    '''
    if (liable_idxBHW is None or liable_num_objects is None): liable_idxBHW, liable_num_objects = parse_idxBHW(y_true, batch_size)
    sum_object = tf.math.reduce_sum(liable_num_objects)
    #    每个批次中不负责检测的cell总数 = 每个批次cell总数 - 每个批次负责检测的cell总数
    fmaps_shape = (yolohard.shape[1], yolohard.shape[2])
    total_objects = tf.ones(shape=(batch_size, ), dtype=tf.int64) * (fmaps_shape[0] * fmaps_shape[1])
    unliable_num_objects = total_objects - liable_num_objects
    unliable_sum_objects = tf.math.reduce_sum(unliable_num_objects)
    
    #    idxBHW扩展为 Tensor(sum_object * num_anchors, 3)
    idxBHW = tf.expand_dims(liable_idxBHW, axis=1)
    idxBHW = tf.repeat(idxBHW, repeats=num_anchors, axis=1)
    idxBHW = tf.reshape(idxBHW, shape=(sum_object * num_anchors, 3))
    #    anchor维的索引 Tensor(sum_object * num_anchors, 1)
    idxA = tf.range(num_anchors, dtype=tf.int64)
    idxA = tf.expand_dims(idxA, axis=0)
    idxA = tf.repeat(idxA, repeats=sum_object, axis=0)
    idxA = tf.reshape(idxA, shape=(sum_object * num_anchors, 1))
    idxBHWA = tf.concat([idxBHW, idxA], axis=-1)
    v = tf.ones(shape=(sum_object * num_anchors,), dtype=tf.int32)
    #    组成系数矩阵
    idx_mat = tf.SparseTensor(indices=idxBHWA, values=v, dense_shape=(batch_size, yolohard.shape[1], yolohard.shape[2], num_anchors))
    idx_mat = tf.sparse.reorder(idx_mat)
    idx_mat = tf.sparse.to_dense(idx_mat)
    
    #    用系数矩阵当索引，找出其中值为0的索引就是不负责预测的anchors的索引
    idx_unliable = tf.where(tf.equal(idx_mat, 0))
    unliable_anchors = tf.gather_nd(yolohard, indices=idx_unliable)
    unliable_anchors = unliable_anchors[:, num_classes]
    unliable_anchors = tf.reshape(unliable_anchors, shape=(unliable_sum_objects, num_anchors, ))
    return unliable_anchors, unliable_num_objects



