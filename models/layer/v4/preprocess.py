# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年2月26日
'''
import tensorflow as tf
import threading

import utils.alphabet as alphabet
import utils.conf as conf
from models.layer.commons.part import YoloHardRegister
from utils.iou import ciou_n2n_tf_ragged, iou_n2n_tf_ragged


#    根据y_true的idxH, idxW从yolohard中拿负责预测的cell（cell中包含anchor信息）
def takeout_liable_cells(yolohard, y_true, batch_size=conf.DATASET_CELLS.get_batch_size()):
    '''根据y_true的idxH, idxW从yolohard中拿负责预测的cell（cell中包含anchor信息）
        @param yolohard: 模型产出的预测 Tensor(batch_size, H, W, num_anchors, num_classes+5)
        @param y_true: 与yolohard对应的y_true Tensor(batch_size, 6, 2 + 5 + num_anchor * 3)
                                                        2: idxH, idxW (-1,-1)代表是填充的值
                                                        5: x,y,w,h, idxV
                                                        num_anchor*3: IoU, w, h
        @param batch_size: 当初设置的批量大小
        @return: liable_cells, y_true_liable, rows_divide
                    liable_cells：负责预测的cells信息（包含预测anchors信息）
                                    RaggedTensor(batch_size, num_object, num_anchors, num_classes+5)
                                                    batch_size: 批量大小
                                                    num_object: 图片物体个数。每个图片可能不一样
                                                    num_anchors: cell中anchor个数
                                                        num_classes+5: 分类个数 + anchor信息
                                                            num_classes: 分类个数
                                                            5: 预测置信度, dx, dy, dw, dh
                    y_true_liables：负责预测的cells信息与其对应的y_true（包含gt信息）
                                    RaggedTensor(batch_size, num_object, 2 + 5 + num_anchor * 3)
                                                    batch_size: 批量大小
                                                    num_object: 图片物体个数。每个图片可能不一样
                                                    2 + 5 + num_anchor * 3: cell坐标信息，标注框信息，每个anchor信息
                                                        2: idxH, idxW
                                                        5: x,y(相对cell左上角坐标), w,h(整图占比), idxV(分类索引)
                                                        num_anchor * 3: 每个anchor的IoU, w, h
                    num_objects：每个batch_size中实际物体数量 tensor(batch_size, )
    '''
    #    取(idxH, idxW)，含有(-1, -1)填充信息
    idx_liable_cells = y_true[:, :, :2]
    idx_liable_cells = tf.cast(idx_liable_cells, dtype=tf.int32)
    #    追加batch_size维度索引，为了下面的gather_no
    idx_batch_size = tf.range(batch_size)
    idx_batch_size = tf.expand_dims(idx_batch_size, axis=-1)
    idx_batch_size = tf.repeat(idx_batch_size, repeats=6, axis=0)
    idx_batch_size = tf.reshape(idx_batch_size, shape=(batch_size, 6, 1))
    #    组成index
    idx = tf.concat([idx_batch_size, idx_liable_cells], axis=-1)
    #    每个batch_size中有效数据个数（剔除-1后的个数）
    num_object = tf.math.count_nonzero(idx[:,:,1] >= 0, axis=1)
    idx = tf.reshape(idx, shape=(idx.shape[0] * idx.shape[1], 3))
    idx = tf.gather_nd(idx, indices=tf.where(idx[:,1] >= 0))
    
    #    实际负责预测的cells RaggedTensor(batch_size, None, 2 + 5 + num_anchor * 3)
    liable_cells = tf.gather_nd(yolohard, indices=idx)
    liable_cells = tf.RaggedTensor.from_row_lengths(liable_cells, row_lengths=num_object)
    #    实际负责预测的cells对应的y_true RaggedTensor(batch_size, None, 2 + 5 + num_anchor * 3)
    y_true_liables = tf.gather_nd(y_true, indices=tf.where(y_true[:, :, 0] >= 0))
    y_true_liables = tf.RaggedTensor.from_row_lengths(y_true_liables, row_lengths=num_object)
    
    return liable_cells, y_true_liables, num_object


#    从liable_cells中取出负责预测物体的anchors
def takeout_liable_anchors(liable_cells, 
                           y_true_liables, 
                           fmaps_shape, 
                           num_classes=len(alphabet.ALPHABET), 
                           num_anchors=conf.DATASET_CELLS.get_anchors_set().shape[1],
                           threshold_liable_iou=conf.V4.get_threshold_liable_iou()):
    '''从liable_cells中取出负责预测物体和不负责预测物体的anchor
        并且追加loss中需要用到的全部值: 
            对于负责的anchor：追加置信度，CIoU, 物体真实分类索引
            对于不负责的anchor：追加置信度
        @param liable_cells: 负责预测的cells信息（包含预测anchors信息）
                             RaggedTensor(batch_size, None(num_object), num_anchors, num_classes + 5)
                                                    batch_size: 批量大小
                                                    num_object: 图片物体个数。每个图片可能不一样
                                                    num_anchors: cell中anchor个数
                                                    num_anchors: cell中anchor个数
                                                        num_classes + 5: 分类个数 + anchor信息
                                                            num_classes: 分类个数
                                                            5: 预测置信度, dx, dy, dw, dh
        @param y_true_liables：负责预测的cells信息与其对应的y_true（包含gt信息）
                             RaggedTensor(batch_size, None(num_object), 2 + 5 + num_anchor * 3)
                                                    batch_size: 批量大小
                                                    num_object: 图片物体个数。每个图片可能不一样
                                                    2 + 5 + num_anchor * 3: cell索引 + gt信息 + 每个anchor信息
                                                        2: idxH, idxW
                                                        5: x,y,w,h, idxV
                                                        num_anchor*3: IoU, w, h
        @param fmaps_shape: tuple(H, W) 此时的特征图高宽
        @param threshold_liable_iou: 负责预测的anchor与gt的IoU阈值。超过阈值的anchor都会被判定为负责预测的anchor
                                        若所有anchor与gt的IoU都小于阈值，则取最大的
        
        @return liable_anchors RaggedTensor(batch_size, num_object, num_liable, num_classes + 7 + 6)
                                            batch_size: 图片批量个数
                                            num_object: 实际物体数，每张图片可能不一样
                                            num_liable: 负责预测的anchor数，每个物体可能不一样(0,3]
                                            num_classes + 7 + 6: 分类数 + anchor信息 + gt信息
                                                num_classes: 各个分类得分
                                                7: anchor的[xl,yl, xr,yr], anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, 
                                                6: gt的[xl,yl, xr,yr, relative_area, idxV]
    '''
    #    找出每个batch_size中，每个物体的信息 RaggedTensor(batch_size, num_object(不固定), 5)
    gt_boxes = get_gt_from_y(y_true_liables, fmaps_shape)
    
    #    找出每个cells中的anchor信息，并还原为anchor_boxes RaggedTensor(batch_size, num_object(不固定), num_anchors, 4)
    anchors_boxes = restore_anchors_boxes(y_true_liables, liable_cells, fmaps_shape, num_anchors, num_classes)
    
    #    计算每个anchor_box与对应gt_boxes的IoU RaggedTensor(batch_size, num_object(不固定), num_anchors, 1)
    gt_boxes_expand = tf.expand_dims(gt_boxes, axis=-2)
    gt_boxes_expand = tf.tile(gt_boxes_expand, multiples=[1, 1, num_anchors, 1])                #    整形为可与anchor_boxes计算格式 RaggedTensor(batch_size, num_object(不固定), 5)
    iou = iou_n2n_tf_ragged(anchors_boxes, gt_boxes_expand)        
    
    #    通过IoU取实际负责预测的anchors 
    #    liable_anchors: RaggedTensor(batch_size * num_object(不固定) * num_anchors_liable(不固定), num_classes+5)
    #    num_objects_every_batch: 每个batch_size中实际物体数量 Tensor(num, )
    #    num_anchors_every_cell: 每个cell实际负责预测的anchor数量，与上面的num_objects联合起来使用 Tensor(num, )
    #    idx_liable_anchors: 符合条件的anchor的索引 Tensor(sum_anchors_liable, 3)
    #                                                         3: idx_batch_size, idx_num_objects, idx_num_anchors
    liable_anchors, num_objects_every_batch, num_anchors_every_cell, idx_liable_anchors = get_liable_anchors(liable_cells, iou, threshold_liable_iou, num_anchors)
    
    #    为实际负责预测的achors追加信息 RaggedTensor(batch_size, num_object(不固定), num_anchors_liable(不固定), num_classes + 7 + 6)
    liable_anchors = append_anchors_info(liable_anchors, iou, gt_boxes_expand, anchors_boxes, num_objects_every_batch, num_anchors_every_cell, idx_liable_anchors, num_classes)
    
    return liable_anchors


#    从y中拿物体信息 tensor(num_object) [x,y(相对于cell左上点偏移), w,h(相对于整图占比), idxV]
def get_gt_from_y(y_true_liable, fmaps_shape):
    '''
        @param y_true_liable: 标签信息 RaggedTensor(batch_size, num_object, 2 + 5 + num_anchor * 3)
                                                    batch_size: 批量大小
                                                    num_object: 图片物体个数。每个图片可能不一样
                                                    2 + 5 + num_anchor * 3: cell坐标信息，标注框信息，每个anchor信息
                                                        2: idxH, idxW
                                                        5: x,y(相对cell左上角坐标), w,h(整图占比), idxV(分类索引)
                                                        num_anchor * 3: 每个anchor的IoU, w, h
        @param idxHW: gt所在cell相对特征图的H,W索引 tensor(num_object, 2) [idxH, idxW]
        @param fmaps_shape: 特征图尺寸
        @return: RaggedTensor(batch_size, num_object, 6) 
                                batch_size: 批量大小
                                num_object: 图片物体个数。每个图片可能不一样
                                6: [xl,yl, xr,yr, relative_area, idxV]
                                    xl,yl, xr,yr: 相对特征图坐标
                                    relative_area: 相对面积（用整图占比的宽高求出的面积。(0,1)之间）
                                    idxV: gt的分类索引
    '''
    #    gt的坐标信息 RaggedTensor(batch_size, num_object, 5)
    gt = y_true_liable[:, :, 2:7]
    #    gt所在的cell坐标 RaggedTensor(batch_size, num_object, 2)
    gt_idxHW = y_true_liable[:, :, :2]
    
    gt_x = tf.math.add(gt[:, :, 0:1] + 0.5, gt_idxHW[:, :, 1:2])              #    相对特征图中心点坐标x    ReggedTensor(batch_size, num_object, 1)
    gt_y = tf.math.add(gt[:, :, 1:2] + 0.5, gt_idxHW[:, :, 0:1])              #    相对特征图中心点坐标y    ReggedTensor(batch_size, num_object, 1)
    gt_w = tf.math.multiply(gt[:, :, 2:3], fmaps_shape[1])                    #    相对特征图宽度          ReggedTensor(batch_size, num_object, 1)
    gt_h = tf.math.multiply(gt[:, :, 3:4], fmaps_shape[0])                    #    相对特征图高度          ReggedTensor(batch_size, num_object, 1)
    gt_xl = tf.math.subtract(gt_x, tf.math.divide(gt_w, 2))                   #    相对特征图左上点坐标x    ReggedTensor(batch_size, num_object, 1)
    gt_yl = tf.math.subtract(gt_y, tf.math.divide(gt_h, 2))                   #    相对特征图左上点坐标y    ReggedTensor(batch_size, num_object, 1)
    gt_xr = tf.math.add(gt_x, tf.math.divide(gt_w, 2))                        #    相对特征图右下点坐标x    ReggedTensor(batch_size, num_object, 1)
    gt_yr = tf.math.add(gt_y, tf.math.divide(gt_h, 2))                        #    相对特征图右下点坐标y    ReggedTensor(batch_size, num_object, 1)
    gt_relative_area = tf.math.multiply(gt[:, :, 2:3], gt[:, :, 3:4])         #    相对面积, 宽高占比求出   ReggedTensor(batch_size, num_object, 1)
    gt_idxV = gt[:, :, 4:5]                                                   #    分类索引               ReggedTensor(batch_size, num_object, 1)
    #    RaggedTensor(batch_size, num_object, 6)
    gts_box = tf.concat([gt_xl,
                         gt_yl,
                         gt_xr,
                         gt_yr,
                         gt_relative_area,
                         gt_idxV], axis=-1)
    return gts_box

#    将anchors还原为anchors_boxes
def restore_anchors_boxes(y_true_liables, liable_cells, fmaps_shape, 
                          num_anchors=conf.DATASET_CELLS.get_anchors_set().shape[1], 
                          num_classes=len(alphabet.ALPHABET)):
    '''
        @param y_true_liables: RaggedTensor(batch_size, num_object, 2 + 5 + num_anchor * 3)
                                                    batch_size: 批量大小
                                                    num_object: 图片物体个数。每个图片可能不一样
                                                    2 + 5 + num_anchor * 3: cell坐标信息，标注框信息，每个anchor信息
                                                        2: idxH, idxW
                                                        5: x,y(相对cell左上角坐标), w,h(整图占比), idxV(分类索引)
                                                        num_anchor * 3: 每个anchor的IoU, w, h
        @param liable_cells：负责预测的cells信息（包含预测anchors信息）
                                    RaggedTensor(batch_size, num_object, num_anchors, num_classes+5)
                                                    batch_size: 批量大小
                                                    num_object: 图片物体个数。每个图片可能不一样
                                                    num_anchors: cell中anchor个数
                                                        num_classes+5: 分类个数 + anchor信息
                                                            num_classes: 分类个数
                                                            5: 预测置信度, dx, dy, dw, dh
        @param fmaps_shape: 特征图高宽信息 [H, W]
        @param num_anchors: 每个cell中的anchor数量
        @param num_classes: 总分类数
        @return: tensor(num_object, num_anchors, 4) [xl,yl, xr,yr]相对特征图坐标
    '''
    #    从y中拿anchor所在cell的左上角坐标，还原为anchor中心点坐标 RaggedTensor(batch_size, num_object, num_anchor, 1)
    cell_lx = y_true_liables[:, :, 1:2]
    cell_lx = tf.expand_dims(cell_lx, axis=-2)
    cell_lx = tf.tile(cell_lx, multiples=[1, 1, num_anchors, 1])                            
    cell_ly = y_true_liables[:, :, 0:1]
    cell_ly = tf.expand_dims(cell_ly, axis=-2)
    cell_ly = tf.tile(cell_ly, multiples=[1, 1, num_anchors, 1])                           
    #    从y中拿anchor的宽高信息 RaggedTensor(batch_size, num_object, num_anchors, 1)
    anchors_w = tf.math.multiply(y_true_liables[:, :, 8::3], fmaps_shape[1])
    anchors_w = tf.expand_dims(anchors_w, axis=-1)
    anchors_h = tf.math.multiply(y_true_liables[:, :, 9::3], fmaps_shape[0])
    anchors_h = tf.expand_dims(anchors_h, axis=-1)
    #    从liable_cells中拿anchor的预测dx,dy, dw,dh RaggedTenspr(batch_size, num_object, num_anchors, 1)
    anchors_dx = liable_cells[:, :, :, num_classes + 1:num_classes + 2]
    anchors_dy = liable_cells[:, :, :, num_classes + 2:num_classes + 3]
    anchors_dw = liable_cells[:, :, :, num_classes + 3:num_classes + 4]
    anchors_dh = liable_cells[:, :, :, num_classes + 4:num_classes + 5]
    #    根据预测信息还原anchor_boxes的xc, yc, w, h
    anchor_boxes_xc = anchors_dx + cell_lx
    anchor_boxes_yc = anchors_dy + cell_ly
    anchor_boxes_w = tf.math.exp(anchors_dw) * anchors_w
    anchor_boxes_h = tf.math.exp(anchors_dh) * anchors_h
    #    还原预测的anchor_boxes RaggedTensor(batch_size, num_object, num_anchors, 4)
    harf_anchor_w = anchor_boxes_w / 2.
    harf_anchor_h = anchor_boxes_h / 2.
    anchor_boxes_lx = anchor_boxes_xc - harf_anchor_w
    anchor_boxes_ly = anchor_boxes_yc - harf_anchor_h
    anchor_boxes_rx = anchor_boxes_xc + harf_anchor_w
    anchor_boxes_ry = anchor_boxes_yc + harf_anchor_h
    anchors_boxes = tf.concat([anchor_boxes_lx, anchor_boxes_ly, anchor_boxes_rx, anchor_boxes_ry], axis=-1)
    
    return anchors_boxes

#    取负责预测的anchors
def get_liable_anchors(liable_cells, 
                       iou,
                       threshold_liable_iou=conf.V4.get_threshold_liable_iou(),
                       num_anchors=conf.DATASET_CELLS.get_anchors_set().shape[1]):
    '''
        @param liable_cells: 负责预测的cells信息（包含预测anchors信息）
                             RaggedTensor(batch_size, None(num_object), num_anchors, num_classes + 5)
                                                    batch_size: 批量大小
                                                    num_object: 图片物体个数。每个图片可能不一样
                                                    num_anchors: cell中anchor个数
                                                    num_anchors: cell中anchor个数
                                                    num_classes+5: 分类个数 + anchor信息
                                                        num_classes: 分类个数
                                                        5: 预测置信度, dx, dy, dw, dh
        @param iou: RaggedTensor(batch_size, num_object(不固定), num_anchors, 1)
        @param threshold_liable_iou: 负责预测的anchor与gt的IoU阈值。超过阈值的anchor都会被判定为负责预测的anchor
                                        若所有anchor与gt的IoU都小于阈值，则取最大的
        @param num_anchors: 每个cell中含有的anchor个数
        @return: liable_anchors: 负责预测的anchors RaggedTensor(batch_size, num_object(不固定), num_anchors_liable(不固定), num_classes+5)
                 num_objects_every_batch: 每个batch_size中实际物体数量 Tensor(num, )
                 num_anchors_every_cell: 每个cell实际负责预测的anchor数量，与上面的num_objects联合起来使用 Tensor(num, )
                 idx_liable_anchors: 符合条件的anchor的索引 Tensor(sum_anchors_liable, 3)
                                                             3: idx_batch_size, idx_num_objects, idx_num_anchors
    '''
    #    取最大IoU的值
    iou_max = tf.math.reduce_max(iou, axis=2)
    iou_max = tf.expand_dims(iou_max, axis=-2)
    iou_max = tf.tile(iou_max, [1, 1, num_anchors, 1])
    #    标记小于阈值但是在cell中最大IoU位置的元素为1，其他为0。
    #    与原IoU相加，保证原cell中若所有IoU都小于阈值，则原最大IoU+1后一定会超过阈值。再用阈值做过滤就能取到原最大IoU
    iou_max_add = tf.where(tf.logical_and(iou < threshold_liable_iou, iou == iou_max), tf.ones_like(iou), tf.zeros_like(iou))
    iou = iou + iou_max_add
    iou_condition = iou >= threshold_liable_iou
    iou_condition = iou_condition.to_tensor(default_value=False)
    
    #    取每个batch_size中物体个数，每个cell中实际负责的anchor个数
    num_anchors_every_cell = tf.math.count_nonzero(iou_condition, axis=2)
    num_objects_every_batch = tf.math.count_nonzero(num_anchors_every_cell, axis=1)
    #    每个batch_size中实际的物体个数 tensor(sum_objects, )
    num_objects_every_batch = tf.reshape(num_objects_every_batch, shape=(num_objects_every_batch.shape[0] * num_objects_every_batch.shape[1]))
    num_anchors_every_cell = tf.reshape(num_anchors_every_cell, shape=(num_anchors_every_cell.shape[0] * num_anchors_every_cell.shape[1]))
    #    每个cell中，IoU超过阈值的anchor个数（实际负责的anchor个数） tensor(sum_anchors_liable, )
    num_anchors_every_cell = num_anchors_every_cell[num_anchors_every_cell > 0]
    
    #    通过IoU阈值取符合条件的anchor
    idx_iou = tf.where(iou >= threshold_liable_iou)
    idx_liable_anchors = idx_iou[:, :3]
    liable_anchors = tf.gather_nd(liable_cells, indices=idx_liable_anchors)                                                                         #    RaggedTensor(sum_anchors_liable, 6)
    liable_anchors = tf.RaggedTensor.from_nested_row_lengths(liable_anchors, nested_row_lengths=[num_objects_every_batch, num_anchors_every_cell])  #    切第一层 RaggedTensor(sum_objects, num_anchors_liable(不固定), 6)
    return liable_anchors, num_objects_every_batch, num_anchors_every_cell, idx_liable_anchors


#    追加后面计算需要用到的值
def append_anchors_info(liable_anchors,
                        iou,
                        gt_boxes_expand,
                        anchors_boxes,
                        num_objects_every_batch,
                        num_anchors_every_cell,
                        idx_liable_anchors,
                        num_classes=len(alphabet.ALPHABET)):
    '''
        @param liable_anchors: 负责预测的cells信息（包含预测anchors信息）
                                 RaggedTensor(batch_size, num_object, num_anchors, num_classes + 5)
                                                        batch_size: 批量大小
                                                        num_object: 图片物体个数。每个图片可能不一样
                                                        num_anchors: cell中anchor个数。每个cell可能不一样
                                                        num_classes+5: 分类个数 + anchor信息
                                                            num_classes: 分类个数
                                                            5: 预测置信度, dx, dy, dw, dh
        @param iou: 负责预测的cells中全部anchor与cell中的gt的IoU RaggedTensor(batch_size, num_object(不固定), num_anchors, 1)
                    通过idx_liable_anchors可取到负责预测的anchor的IoU
        @param gt_boxes_expand: 每个batch_size中的物体信息 RaggedTensor(batch_size, num_object(不固定), num_anchors, 6)
                                                            6: [xl,yl, xr,yr, relative_area, idxV]
                                                                    xl,yl, xr,yr: 相对特征图坐标
                                                                    relative_area: 相对面积（用整图占比的宽高求出的面积。(0,1)之间）
                                                                    idxV: gt的分类索引
        @param anchors_boxes: 每个cells中的anchor信息 RaggedTensor(batch_size, num_object(不固定), num_anchors, 4): 
                                                    4: xl,yl, xr,yr: 相对特征图坐标
        @param num_objects_every_batch: 每个batch_size中实际物体数量 Tensor(num, )
        @param num_anchors_every_cell: 每个cell实际负责预测的anchor数量，与上面的num_objects联合起来使用 Tensor(num, )
        @param idx_liable_anchors: 符合条件的anchor的索引 Tensor(sum_anchors_liable, 3)
                                                             3: idx_batch_size, idx_num_objects, idx_num_anchors
        @return liable_anchors RaggedTensor(batch_size, num_object, num_liable, num_classes + 7 + 6)
                                            batch_size: 图片批量个数
                                            num_object: 实际物体数，每张图片可能不一样
                                            num_liable: 负责预测的anchor数，每个物体可能不一样(0,3]
                                            num_classes + 7 + 6: 分类数 + anchor信息 + gt信息
                                                num_classes: 各个分类得分
                                                7: anchor的[xl,yl, xr,yr], anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, 
                                                6: gt的[xl,yl, xr,yr, relative_area, idxV]
    '''
    #    取liable_anchors中预测的各分类得分 RaggedTensor(batch_size, num_object(不固定), num_liable_anchors(不固定), 1)
    liable_anchors_classes = liable_anchors[:, :, :, :num_classes]
    
    #    取liable_anchors中预测的置信度 RaggedTensor(batch_size, num_object(不固定), num_liable_anchors(不固定), 1)
    liable_anchors_confidence = liable_anchors[:, :, :, num_classes:num_classes+1]
    
    #    取liable_anchors对应的[lx,ly, rx,ry] RaggedTensor(batch_size, num_object(不固定), num_liable_anchors(不固定), 4)
    liable_anchors_boxes = liable_anchors[:, :, :, num_classes+1:num_classes+1+4]
    
    #    计算anchors_boxes与gt_boxes_expands的CIoU（挑出liable后有两个不固定的维度，目前还不知道怎么广播）
    #    RaggedTensor(batch_size, num_object(不固定), num_anchors, 1) 
    ciou = ciou_n2n_tf_ragged(anchors_boxes, gt_boxes_expand, iou)
    
    #    取liable_anchors标记的置信度(其实就是gt_boxes中对应的IoU) RaggedTensor(batch_size, num_object(不固定), num_liable_anchors(不固定), 1)
    #    iou
    
    #    取gt_boxes中负责预测的anchors对应的gt信息(其实就是gt_boxes_expand)  RaggedTensor(batch_size * num_object(不固定) * num_anchors, 6)
    #    gt_boxes_expand
    
    #    concat: iou + ciou + gt_info，并整合成: RaggedTensor(batch_size, num_object(不固定), num_liable_anchors(不固定), 8)
    iou_ciou_gt = tf.concat([iou, ciou, gt_boxes_expand], axis=-1)
    iou_ciou_gt = tf.gather_nd(iou_ciou_gt, indices=idx_liable_anchors)
    iou_ciou_gt = tf.RaggedTensor.from_nested_row_lengths(iou_ciou_gt, nested_row_lengths=[num_objects_every_batch, num_anchors_every_cell])
    
    #    整合数据
    #    liable_anchors_classes + liable_anchors_boxes + liable_anchors_confidence + iou_ciou_gt
    liable_anchors = tf.concat([liable_anchors_classes, 
                                liable_anchors_boxes, 
                                liable_anchors_confidence, 
                                iou_ciou_gt], axis=-1)
    return liable_anchors


#    取特征图中不负责预测的anchors
def takeout_unliable_anchors(y_true, 
                             yolohard, 
                             batch_size=conf.DATASET_CELLS.get_batch_size(),
                             num_anchors=conf.DATASET_CELLS.get_anchors_set().shape[1],
                             num_classes=len(alphabet.ALPHABET)):
    '''取特征图中无目标的anchors，数量较大
        @param y_true：与yolohard对应的y_true Tensor(batch_size, 6, 2 + 5 + num_anchor * 3)
                                                    2: idxH, idxW (-1,-1)代表是填充的值
                                                    5: x,y,w,h, idxV
                                                    num_anchor*3: IoU, w, h
        @param yolohard: 模型产出的预测 tensor(batch_size, H, W, num_anchors, num_classes+5)
        @param num_anchor: 每个cell中anchor数
        @return Tensor(batch_size, H, W, num_anchors)
                        
    '''
    #    所有负责预测物体的cell坐标 Tensor(batch_size, num_object, 2)
    #    idxHW中含有(-1, -1)的填充值，后面会利用此值去掉填充 Tensor(batch_size, H, W, 2)
    idxHW = y_true[:, :, :2]
    idxHW = tf.cast(idxHW, dtype=tf.int32)
    #    batch_size维的索引 Tensor(batch_size, H, W, 1)
    idxB = tf.range(batch_size)
    idxB = tf.expand_dims(idxB, axis=-1)
    idxB = tf.repeat(idxB, repeats=idxHW.shape[1], axis=0)
    idxB = tf.reshape(idxB, shape=(batch_size, idxHW.shape[1], 1))
    #    B,H,W维的索引 Tensor(batch_size, H, W, 3)
    idxBHW = tf.concat([idxB, idxHW], axis=-1)
    #    anchor维的索引 Tensor(batch_size, H, W, num_anchor, 1)
    idxA = tf.range(num_anchors)
    idxA = tf.expand_dims(idxA, axis=0)
    idxA = tf.repeat(idxA, repeats=idxHW.shape[1], axis=0)
    idxA = tf.repeat(tf.expand_dims(idxA, axis=0), repeats=batch_size, axis=0)
    idxA = tf.expand_dims(idxA, axis=-1)
    #    idxBHW扩充为 Tensor(batch_size, H, W, num_anchor, 3)
    idxBHW = tf.expand_dims(idxBHW, axis=-2)
    idxBHW = tf.repeat(idxBHW, repeats=num_anchors, axis=-2)
    #    idxA与idxBHW合并，并去除idxHW=(-1, -1)的填充值 Tensor(num_un_liable_anchors, 4) [idx_batch_idx, idx_h, idx_w, idx_anchor]
    idxBHWA = tf.concat([idxBHW, idxA], axis=-1)
    idxBHWA = tf.cast(idxBHWA[idxBHWA[:, :, :, 1] >= 0], dtype=tf.int64)
    v = tf.ones(shape=(idxBHWA.shape[0]), dtype=tf.int32)
    #    组成稀疏矩阵，负责预测的cell为1，不负责为0
    idx = tf.SparseTensor(indices=idxBHWA, values=v, dense_shape=(batch_size, yolohard.shape[1], yolohard.shape[2], num_anchors))
    idx = tf.sparse.reorder(idx)
    idx = tf.sparse.to_dense(idx)
    #    通过稀疏矩阵的0值位置拿出不负责预测的anchors信息 Tensor(bath_size, H, W, num_anchor)
    yolohard_confidence = yolohard[:, :, :, :, num_classes]
    unliable_anchors_fmaps = tf.where(idx == 0, yolohard_confidence, tf.zeros_like(yolohard_confidence))
    
    return unliable_anchors_fmaps


#    暂存yolohard解析出来的结果
class AnchorsRegister():
    '''暂存yolohard解析出来的结果。“最多跑一次”
        在一个batch中，takeout_liable_cells, takeout_liable_anchors, takeout_unliable_anchors的只跑一次
    '''
    _instance_lock = threading.Lock()
    
    def __init__(self):
        pass
    
    @classmethod
    def instance(cls, *args, **kwargs):
        with AnchorsRegister._instance_lock:
            if not hasattr(AnchorsRegister, '_instance'):
                AnchorsRegister._instance = AnchorsRegister(*args, **kwargs)
            pass
        return AnchorsRegister._instance
    
    #    解析新的结果，并暂存
    def parse_anchors_and_cache(self, yolohard_register=YoloHardRegister.instance(), y_true=None):
        #    从yolohard_register中拿yolohard1, yolohard2, yolohard3
        yolohard1 = yolohard_register.get_yolohard1()               #    (batch_size, 23, 60, num_anchors, num_classes+5)
        y_true1 = y_true[:, 2, :,:]                                 #    对应scale[2]尺寸，最大尺寸（低层特征感受野较小，适合检测小物体）
        fmaps_shape1 = (yolohard1.shape[1], yolohard1.shape[2])
        liable_cells1, y_true_liable1, _ = takeout_liable_cells(yolohard1, y_true1)
        liable_anchors1 = takeout_liable_anchors(liable_cells1, y_true_liable1, fmaps_shape1)
        unliable_anchors1 = takeout_unliable_anchors(y_true1, yolohard1)
        self.cacheYoloHard1(liable_cells1, y_true_liable1, liable_anchors1, unliable_anchors1)
        
        yolohard2 = yolohard_register.get_yolohard2()               #    (batch_size, 12, 30, num_anchors, num_classes+5)
        y_true2 = y_true[:, 1, :,:]                                 #    对应scale[1]尺寸，中等尺寸
        fmaps_shape2 = (yolohard2.shape[1], yolohard2.shape[2])
        liable_cells2, y_true_liable2, _ = takeout_liable_cells(yolohard2, y_true2)
        liable_anchors2 = takeout_liable_anchors(liable_cells2, y_true_liable2, fmaps_shape2)
        unliable_anchors2 = takeout_unliable_anchors(y_true2, yolohard2)
        self.cacheYoloHard2(liable_cells2, y_true_liable2, liable_anchors2, unliable_anchors2)
        
        yolohard3 = yolohard_register.get_yolohard3()               #    (batch_size, 6,  15, num_anchors, num_classes+5)
        y_true3 = y_true[:, 0, :,:]                                 #    对应scale[0]尺寸，最小尺寸（高层特征感受野较大，适合检测大物体）
        fmaps_shape3 = (yolohard3.shape[1], yolohard3.shape[2])
        liable_cells3, y_true_liable3, _ = takeout_liable_cells(yolohard3, y_true3)
        liable_anchors3 = takeout_liable_anchors(liable_cells3, y_true_liable3, fmaps_shape3)
        unliable_anchors3 = takeout_unliable_anchors(y_true3, yolohard3)
        self.cacheYoloHard3(liable_cells3, y_true_liable3, liable_anchors3, unliable_anchors3)
        
        return (liable_anchors1, unliable_anchors1), \
                (liable_anchors2, unliable_anchors2), \
                (liable_anchors3, unliable_anchors3)
    
    #    暂存yolohard解析结果
    def cacheYoloHard1(self, liable_cells, y_true_liable, liable_anchors, unliable_anchors):
        self._liable_cells1 = liable_cells
        self._y_true_liable1 = y_true_liable
        self._liable_anchors1 = liable_anchors
        self._unliable_anchors1 = unliable_anchors
        pass
    def cacheYoloHard2(self, liable_cells, y_true_liable, liable_anchors, unliable_anchors):
        self._liable_cells2 = liable_cells
        self._y_true_liable2 = y_true_liable
        self._liable_anchors2 = liable_anchors
        self._unliable_anchors2 = unliable_anchors
        pass
    def cacheYoloHard3(self, liable_cells, y_true_liable, liable_anchors, unliable_anchors):
        self._liable_cells3 = liable_cells
        self._y_true_liable3 = y_true_liable
        self._liable_anchors3 = liable_anchors
        self._unliable_anchors3 = unliable_anchors
        pass
    #    取缓存
    def get_yolohard1(self):
        return self._liable_cells1, self._y_true_liable1, self._liable_anchors1, self._unliable_anchors1
    def get_yolohard2(self):
        return self._liable_cells2, self._y_true_liable2, self._liable_anchors2, self._unliable_anchors2
    def get_yolohard3(self):
        return self._liable_cells3, self._y_true_liable3, self._liable_anchors3, self._unliable_anchors3
    
    pass