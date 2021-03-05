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
from utils.iou import ciou_n21_tf, iou_b_n21_tf


#    根据y_true的idxH, idxW从yolohard中拿负责预测的cell（cell中包含anchor信息）
def takeout_liable_cells(yolohard, y_true):
    '''根据y_true的idxH, idxW从yolohard中拿负责预测的cell（cell中包含anchor信息）
        @param yolohard: 模型产出的预测 tensor(batch_size, H, W, num_anchors, num_classes+5)
        @param y_true: 与yolohard对应的y_true tensor(batch_size, 6, 2 + 5 + num_anchor * 3)
        @return: liable_cells, y_true_liable
                    liable_cells：负责预测的cells信息（包含预测anchors信息）
                        list [batch_size个]
                            tensor (物体个数, num_anchors, num_classes+5)
                    y_true_liable：负责预测的cells信息与其对应的y_true（包含gt信息）
                        list [batch_size个]
                            tensor (物体个数, 2 + 5 + num_anchor * 3)
                    两个列表的物体个数一一对应
                    
    '''
    #    找到本尺寸下的y_true tensor(batch_size, 6, 2 + 5 + num_anchor * 3)
    #    批量大小
    B = tf.math.count_nonzero(y_true[:,0,0] + 1)

    #    到每个batch_size里单独拿数据。1个batch_size代表1张图片，不同的图片物体数量是不同的，所以不能组成张量
    yolohard_liable_cells = []
    y_true_liable = []
    for b in range(B):
        y_true_b = y_true[b]
        yolohard_b = yolohard[b]
        #    去掉追加的全-1的值
        condition_h = y_true_b[:,0] >= 0
        condition_w = y_true_b[:,1] >= 0
        condition = tf.logical_and(condition_h, condition_w)
        indices = tf.where(condition)
        y_true_b = tf.gather_nd(y_true_b, indices=indices)
        y_true_liable.append(y_true_b)
        
        #    通过idxH, idxW取cell信息
        idxH = y_true_b[:,0]
        idxW = y_true_b[:,1]
        idxHW = tf.concat([tf.expand_dims(idxH, axis=-1),
                           tf.expand_dims(idxW, axis=-1)], axis=-1)
        idxHW = tf.cast(idxHW, dtype=tf.int32)
        cells = tf.gather_nd(yolohard_b, indices=idxHW)
        yolohard_liable_cells.append(cells)
        pass
        
    return yolohard_liable_cells, y_true_liable


#    从liable_cells中取出负责预测物体的anchors
def takeout_liable_anchors(liable_cells, 
                           y_true_liable, 
                           fmaps_shape, 
                           num_classes=len(alphabet.ALPHABET), 
                           num_anchors=conf.DATASET_CELLS.get_anchors_set().shape[1],
                           threshold_liable_iou=conf.V4.get_threshold_liable_iou()):
    '''从liable_cells中取出负责预测物体和不负责预测物体的anchor
        并且追加loss中需要用到的全部值: 
            对于负责的anchor：追加置信度，CIoU, 物体真实分类索引
            对于不负责的anchor：追加置信度
        @param liable_cells: 负责预测的cells信息（包含预测anchors信息）
                             list [batch_size个]
                                tensor (物体个数, num_anchors, num_classes+5) 
                                        num_classes: 各分类得分
                                        5: [confidence, dx,dy(相对cell左上角的偏移),dw,dh(图片缩放比)]
        @param y_true_liable：负责预测的cells信息与其对应的y_true（包含gt信息）
                             list [batch_size个]
                                tensor (物体个数, 2 + 5 + num_anchor * 3): 
                                        2: idxH, idxW
                                        5: [x,y(相对cell左上角的偏移), w,h(整图占比), idxV]
                                        num_anchor * 3: num_anchor个anchor的[IoU, w,h(整图占比)]
        @param fmaps_shape: tuple(H, W) 此时的特征图高宽
        @param threshold_liable_iou: 负责预测的anchor与gt的IoU阈值。超过阈值的anchor都会被判定为负责预测的anchor
                                        若所有anchor与gt的IoU都小于阈值，则取最大的
        
        @return: list([ ... batch_size个 ...])
                    list([... num_object个 ...])
                        tensor(num_liable, num_classes + 7 + 6)
                                num_object: 图片中物体个数
                                num_liable: 每个物体所在的cell中负责检测的anchor数
                                num_classes: 各个分类得分
                                7: anchor置信度预测，anchor的置信度标记，anchor与gt的CIoU, anchor的[xl,yl, xr,yr]
                                6: gt的[xl,yl, xr,yr, relative_area, idxV]
                    ...
    '''
    res = []
    #    循环每对cell和y
    for cells, y in zip(liable_cells, y_true_liable):
        #    图片包含的物体总数
        num_object = y.shape[0]
        
        #    从y中拿idxH,idxW信息 tensor(num_object, 2) [idxH, idxW(相对于特征图HW轴索引)]
        idxHW = y[:, 0:2]
        
        #    从y中拿物体信息, 并还原为gt_box 
        #    tensor(num_object, 6) [xl,yl, xr,yr, relative_area, idxV]
        gts_box = get_gt_from_y(y, idxHW, fmaps_shape)
        
        #    从cells中拿anchors信息，并还原为anchors_boxes 
        #    tensor(num_object, num_anchors, 4) [xl,yl, xr,yr]相对特征图坐标
        anchors_boxes = restore_box(y, cells, idxHW, num_anchors, num_classes)
        
        #    取出负责预测的anchors和不负责预测的anchors
        #    负责预测的anchors list[... num_object ...] tensor(num_liable_anchors, num_classes + 5), 索引信息[idx_object, idx_anchors]
        #    不负责预测的anchors list[... num_object ...] tensor(num_liable_anchors - 1, num_classes + 5), 索引信息[idx_object, idx_anchors]
        liable_anchors_list, idx_liables_list = get_liable_anchors(cells, gts_box, anchors_boxes, num_object, num_anchors, threshold_liable_iou)
        
        #    循环每对liable_anchors和他们的索引
        liable_anchors_info_list = []
        for liable_anchors, idx_liables, anchors_box, y_cell, gt_box in zip(liable_anchors_list, idx_liables_list, anchors_boxes, y, gts_box):
            #    liable_anchors tensor(num_liable_anchors, num_class+5)
            #    idx_liables tensor(num_liable_anchors, )
            #    anchors_box tensor(num_anchors, 4)
            #    y_cell tensor(2 + 5 + num_anchor * 3, ) 
            #    gt_box tensor(6,)
            
            #    计算负责预测anchors与其对应的gt的CIoU tensor(num_liable_anchors, 1)
            liable_anchors_box = tf.gather_nd(anchors_box, idx_liables)
            liable_ciou = ciou_n21_tf(liable_anchors_box, gt_box)
            liable_ciou = tf.expand_dims(liable_ciou, axis=-1)
            #    取出负责预测anchors的预测置信度 tensor(num_liable_anchors, 1)
            liable_confidence_prob = liable_anchors[:, num_classes]
            liable_confidence_prob = tf.expand_dims(liable_confidence_prob, axis=-1)
            #    取出负责预测anchors的标记置信度 tensor(num_liable_anchors, 1) 其实就是IoU
            y_anchors_info = y_cell[7:]
            y_anchors_info = tf.reshape(y_anchors_info, shape=(num_anchors, 3))                   #    tensor(num_anchors, 3)
            liable_y = tf.gather_nd(y_anchors_info, indices=idx_liables)                          #    tensor(num_liable_anchors, 3)
            liable_confidence_true = liable_y[:,0]                                                  #    tensor(num_liable_anchors, 1)
            liable_confidence_true = tf.expand_dims(liable_confidence_true, axis=-1)
            #    取出负责预测anchors的各个分类得分
            liable_cls = liable_anchors[:, :num_classes]                                          #    tensor(num_liable_anchors, 5)
            #    当前需要追加的gt_box信息 tensor(num_liable_amchprs, 6)
            gt_box = tf.expand_dims(gt_box, axis=0)
            gt_box = tf.repeat(gt_box, repeats=liable_anchors.shape[0], axis=0)
            #    合并为负责预测anchors信息 tensor(num_liable_classes + 7 + 6)
            liable_anchors_info = tf.concat([#    各个分类得分
                                             liable_cls,
                                             #    预测置信度，真实置信度，CIoU，anchor的[xl,yl, xr,yr]
                                             liable_confidence_prob,  
                                             liable_confidence_true, 
                                             liable_ciou,
                                             liable_anchors_box,
                                             #    gt的[xl,yl, xr,yr, relative_area, idxV]
                                             gt_box,
                                             ], axis=-1)
            liable_anchors_info_list.append(liable_anchors_info)
            pass
        
#         #    计算负责预测anchors与其对应的gt的CIoU tensor(num_object, 1)
#         liable_anchors_box = tf.gather_nd(anchors_boxes, idx_liable)
#         liable_anchors_box = tf.expand_dims(liable_anchors_box, axis=1)
#         liable_ciou = ciou_b_n21_tf(liable_anchors_box, gts_box)
#         #    取出负责预测anchors的预测置信度 tensor(num_object, 1)
#         liable_confidence_prob = liable_anchors[:, num_classes]
#         #    取出负责预测anchors的标记置信度 tensor(num_object, 1) 其实就是IoU
#         y_anchors_info = y[:, 7:]
#         y_anchors_info = tf.reshape(y_anchors_info, shape=(num_object, num_anchors, 3))           #    tensor(num_object, num_anchors, 3)
#         liable_y = tf.gather_nd(y_anchors_info, indices=idx_liable)
#         liable_confidence_true = liable_y[:, 0]
#         #    取出负责预测anchors的各个分类得分
#         liable_cls = liable_anchors[:, :num_classes]
#         #    合并为负责预测anchors信息 tensor(num_object, num_classes + 2 + 2)
#         liable_anchors_info = tf.concat([#    各个分类得分
#                                          liable_cls,
#                                          #    预测置信度，真实置信度，CIoU，anchor的[xl,yl, xr,yr]
#                                          tf.expand_dims(liable_confidence_prob, axis=-1), 
#                                          tf.expand_dims(liable_confidence_true, axis=-1), 
#                                          liable_ciou,
#                                          tf.squeeze(liable_anchors_box),
#                                          #    gt的[xl,yl, xr,yr, relative_area, idxV]
#                                          gts_box,
#                                          ], axis=-1)
        
#         #    取出不负责预测anchors的置信度 tensor(num_object, num_anchors-1, 1)
#         liable_confidence = unliable_anchors[:, num_classes]
#         unliable_anchors_info = tf.expand_dims(liable_confidence, axis=-1)
        
        res.append(liable_anchors_info_list)
        pass
    
    return res


#    从y中拿物体信息 tensor(num_object) [x,y(相对于cell左上点偏移), w,h(相对于整图占比), idxV]
def get_gt_from_y(y, idxHW, fmaps_shape):
    '''
        @param y: 标签信息 tensor(物体个数, 2 + 5 + num_anchor * 3)
        @param idxHW: gt所在cell相对特征图的H,W索引 tensor(num_object, 2) [idxH, idxW]
        @param fmaps_shape: 特征图尺寸
        @return: tensor(num_object, 6) [xl,yl, xr,yr, relative_area, idxV]
                            xl,yl, xr,yr: 相对特征图坐标
                            relative_area: 相对面积（用整图占比的宽高求出的面积。(0,1)之间）
                            idxV: gt的分类索引
    '''
    gt = y[:, 2:7]
    
    gt_x = gt[:, 0] + idxHW[:, 1]
    gt_y = gt[:, 1] + idxHW[:, 0]
    gt_w = gt[:, 2] * fmaps_shape[1]
    gt_h = gt[:, 3] * fmaps_shape[0]
    gt_xl = gt_x - gt_w/2
    gt_yl = gt_y - gt_h/2
    gt_xr = gt_x + gt_w/2
    gt_yr = gt_y + gt_h/2
    gt_area = gt[:, 2] * gt[:, 3]
    gt_idxV = gt[:, 4]
    #    tensor(num_object, 5)
    gts_box = tf.concat([tf.expand_dims(gt_xl, axis=-1),
                         tf.expand_dims(gt_yl, axis=-1),
                         tf.expand_dims(gt_xr, axis=-1),
                         tf.expand_dims(gt_yr, axis=-1),
                         tf.expand_dims(gt_area, axis=-1),
                         tf.expand_dims(gt_idxV, axis=-1)], axis=-1)
    return gts_box

#    将anchors还原为anchors_boxes
def restore_box(y, cells, idxHW, num_anchor, num_classes):
    '''
        @param y: 标签信息 tensor(物体个数, 2 + 5 + num_anchor * 3)
        @param cells: 负责预测的cell信息 tensor(num_object, num_anchors, num_classes+5)
        @param idxHW: cell在特征图中的坐标 tensor(num_object, 2) [idxH, idxW]
        @param num_anchor: 每个cell中anchor个数
        @param num_classes: 总分类数
        @return: tensor(num_object, num_anchors, 4) [xl,yl, xr,yr]相对特征图坐标
    '''
    #    从y中拿anchor的宽高信息 tensor(num_object, num_anchor, 3)  [IoU, w,h(相对于整图占比)]
    anchors_y = y[:, 7:]
    anchors_y = tf.reshape(anchors_y, shape=(anchors_y.shape[0], num_anchor, 3))
    #    从cell拿anchors信息 tensor(num_object, num_anchor, num_classes+5)
    #    似乎不用拿，cell就是完整的
    #    还原预测的box tensor(num_object, num_anchor, 4) [xl,yl, xr,yr]
    anchors_boxes_x = cells[:, :, num_classes + 1] + tf.expand_dims(idxHW[:,1], axis=-1)            #    preb_box(x) = d(x) + cell(x)
    anchors_boxes_y = cells[:, :, num_classes + 2] + tf.expand_dims(idxHW[:,0], axis=-1)            #    preb_box(y) = d(y) + cell(y)
    anchors_boxes_w = tf.math.exp(cells[:, :, num_classes + 3]) * anchors_y[:,:,1]                  #    exp(d(w)) * anchor(w)
    anchors_boxes_h = tf.math.exp(cells[:, :, num_classes + 4]) * anchors_y[:,:,2]                  #    exp(d(h)) * anchor(h)
    anchors_boxes_xl = anchors_boxes_x - anchors_boxes_w/2
    anchors_boxes_yl = anchors_boxes_y - anchors_boxes_h/2
    anchors_boxes_xr = anchors_boxes_x + anchors_boxes_w/2
    anchors_boxes_yr = anchors_boxes_y + anchors_boxes_h/2
    #    tensor(num_object, num_anchors, 4)
    anchors_boxes = tf.concat([tf.expand_dims(anchors_boxes_xl, axis=-1),
                               tf.expand_dims(anchors_boxes_yl, axis=-1),
                               tf.expand_dims(anchors_boxes_xr, axis=-1),
                               tf.expand_dims(anchors_boxes_yr, axis=-1)], axis=-1)
    return anchors_boxes

#    取负责预测的anchors，和cell中不负责预测的anchors
def get_liable_anchors(cells, gts_box, anchors_boxes, num_object, num_anchor, 
                       threshold_liable_iou=conf.V4.get_threshold_liable_iou()):
    '''
        @param cells: 负责预测的cell信息 tensor(num_object, num_anchors, num_classes+5)
        @param gts_box: 标记框还原出的gts_box tensor(num_object, 4) [xl,yl, xr,yr]相对特征图坐标
        @param anchors_boxes: anchors信息还原出的anchors_boxes tensor(num_object, num_anchors, 4) [xl,yl, xr,yr]相对特征图坐标
        @param num_object: 图片中的物体总数
        @param num_anchor: 每个cell中的anchor数
        @param threshold_liable_iou: 负责预测的anchor与gt的IoU阈值。超过阈值的anchor都会被判定为负责预测的anchor
                                        若所有anchor与gt的IoU都小于阈值，则取最大的
        @return: liable_anchors: 负责预测的anchors list[... num_object个 ...]  
                                                    tensor(num_liable, num_classes + 5),
                                                        num_liable: 负责预测的anchor个数
                                                        num_classes: 分类预测
                                                        5: [confidence, dx,dy(相对cell左上角的偏移),dw,dh(图片缩放比)]
                 idx_liable: 负责预测的anchor在cell的anchors中的索引 list[... num_object个 ...]
                                                     tensor(num_liable, 2)
                                                         num_liable: 负责预测的anchor个数
                                                         2: [第几个cell, 第几个anchor]
    '''
    #    计算anchors与gt的IoU tensor(num_object, num_anchor)
    iou = iou_b_n21_tf(rect_srcs=anchors_boxes, rect_tag=gts_box)
    #    取大于阈值的IoU的索引
    idx_iou = tf.where(iou > threshold_liable_iou)
    
    #    如果没有超过阈值的IoU，则取最大IoU
    if (idx_iou.shape[0] == 0):
        idx_max_iou = tf.math.argmax(iou, axis=-1)
        idx_max_iou = tf.cast(idx_max_iou, dtype=tf.int32)
        idxB = tf.range(num_object)
        idx_iou = tf.stack([idxB, idx_max_iou], axis=-1)
        pass
    
    #    组合成list (... num_object ...) tensor(num_liable, num_classes + 5)
    liable_anchors = []
    idx_liable = []
    for o in range(num_object):
        idx_liables = idx_iou[idx_iou[:,0] == o]
        liables = tf.gather_nd(cells, idx_liables)
        liable_anchors.append(liables)
        idx_liables = idx_liables[:,1]
        idx_liables = tf.expand_dims(idx_liables, axis=-1)
        idx_liable.append(idx_liables)
        pass
    
    #    每个num_object取IoU最大的索引，和最大的IoU
#     idx_max_iou = tf.math.argmax(iou, axis=-1)
#     max_iou = tf.math.reduce_max(iou, axis=-1)
    #    取负责预测的anchor
#     idx_max_iou = tf.cast(idx_max_iou, dtype=tf.int32)
#     idxB = tf.range(num_object)
#     idx_liable = tf.stack([idxB, idx_max_iou], axis=-1)
#     liable_anchors = tf.gather_nd(cells, idx_liable)                      #    负责预测的anchors tensor(num_object, num_classes + 5)
        
    #    取不负责预测的anchors
    #    如果max_iou中存在0值（真实数据上不会存在，测试时随机数据中会存在），则加一个很小的值
#     if (tf.math.count_nonzero(max_iou) > 0): 
#         max_iou = tf.where(max_iou > 0, max_iou, tf.ones_like(max_iou))
#         print(max_iou)
#         pass
#     max_iou = tf.expand_dims(max_iou, axis=-1)
#     idx_unliable_incell = tf.where(iou < max_iou)
#     idx_unliable_incell = tf.reshape(idx_unliable_incell, shape=(num_object, num_anchor-1, 2))
#     unliable_anchors_incell = tf.gather_nd(cells, idx_unliable_incell)                  #    不负责预测的anchors tensor(num_object, num_anchors - 1, num_classes + 5)

    return liable_anchors, idx_liable
#     return liable_anchors, idx_liable, unliable_anchors_incell, idx_unliable_incell


#    取特征图中不负责预测的anchors
def takeout_unliable_anchors(y_true_liable, yolohard, num_anchors=conf.DATASET_CELLS.get_anchors_set().shape[1]):
    '''取特征图中无目标的anchors，数量较大
        @param y_true_liable: 负责预测的cells信息与其对应的y_true（包含gt信息）
                             list [batch_size个]
                                tensor (物体个数, 2 + 5 + num_anchor * 3): 
                                        2: idxH, idxW
                                        5: [x,y(相对cell左上角的偏移), w,h(整图占比)]
                                        num_anchor * 3: num_anchor个anchor的[IoU, w,h(整图占比)]
        @param num_anchor: 每个cell中anchor数
        @param yolohard: 模型产出的预测 tensor(batch_size, H, W, num_anchors, num_classes+5)
        return tensor(batch_size, H, W, num_anchors)
                        
    '''
    unliable_anchors_fmaps = []
    for i, y in enumerate(y_true_liable):
        num_object = y.shape[0]
        
        #    每个batch_size的特征图 tensor(H, W, num_anchors, num_classes+5)
        fmaps = yolohard[i]
        fmaps = fmaps[:,:, :, num_anchors]              #    tensor(H, W, num_anchors)
        #    有物体的cell索引 tensor(num_object, 2)
        idxHW = tf.cast(y[:, :2], dtype=tf.int64)
        idxHW = tf.repeat(tf.expand_dims(idxHW, axis=1), repeats=num_anchors, axis=1)
        idxA = tf.expand_dims(tf.range(num_anchors), axis=-1)
        idxA = tf.expand_dims(idxA, axis=0)
        idxA = tf.repeat(idxA, repeats=num_object, axis=0)
        idxA = tf.cast(idxA, dtype=tf.int64)
        idx = tf.concat([idxHW, idxA], axis=-1)
        idx = tf.reshape(idx, shape=(idx.shape[0] * idx.shape[1], idx.shape[-1]))
        
        #    会存在多个物体落入同一个cell的情况，需要对idx去重
        #    目前是数据那边保证同一个scale的同一个cell只能有1个物体
        
        v = [1 for _ in range(num_object * num_anchors)]
        idx = tf.SparseTensor(indices=idx, values=v, dense_shape=[fmaps.shape[0], fmaps.shape[1], num_anchors])
        idx = tf.sparse.reorder(idx)
        idx = tf.sparse.to_dense(idx)
        fmaps = tf.where(idx == 0, fmaps, tf.zeros_like(fmaps))
        unliable_anchors_fmaps.append(fmaps)
        pass
    unliable_anchors_fmaps = tf.convert_to_tensor(unliable_anchors_fmaps)
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
        liable_cells1, y_true_liable1 = takeout_liable_cells(yolohard1, y_true1)
        liable_anchors1 = takeout_liable_anchors(liable_cells1, y_true_liable1, fmaps_shape1)
        unliable_anchors1 = takeout_unliable_anchors(y_true_liable1, yolohard1)
        self.cacheYoloHard1(liable_cells1, y_true_liable1, liable_anchors1, unliable_anchors1)
        
        yolohard2 = yolohard_register.get_yolohard2()               #    (batch_size, 12, 30, num_anchors, num_classes+5)
        y_true2 = y_true[:, 1, :,:]                                 #    对应scale[1]尺寸，中等尺寸
        fmaps_shape2 = (yolohard2.shape[1], yolohard2.shape[2])
        liable_cells2, y_true_liable2 = takeout_liable_cells(yolohard2, y_true2)
        liable_anchors2 = takeout_liable_anchors(liable_cells2, y_true_liable2, fmaps_shape2)
        unliable_anchors2 = takeout_unliable_anchors(y_true_liable2, yolohard2)
        self.cacheYoloHard2(liable_cells2, y_true_liable2, liable_anchors2, unliable_anchors2)
        
        yolohard3 = yolohard_register.get_yolohard3()               #    (batch_size, 6,  15, num_anchors, num_classes+5)
        y_true3 = y_true[:, 0, :,:]                                 #    对应scale[0]尺寸，最小尺寸（高层特征感受野较大，适合检测大物体）
        fmaps_shape3 = (yolohard3.shape[1], yolohard3.shape[2])
        liable_cells3, y_true_liable3 = takeout_liable_cells(yolohard3, y_true3)
        liable_anchors3 = takeout_liable_anchors(liable_cells3, y_true_liable3, fmaps_shape3)
        unliable_anchors3 = takeout_unliable_anchors(y_true_liable3, yolohard3)
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