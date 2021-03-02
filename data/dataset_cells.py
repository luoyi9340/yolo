# -*- coding: utf-8 -*-  
'''
网格数据生成器
    1 根据特征图缩放比例计算每张图片生成网格数据
        特征图缩放比例可能会有多个
    2 检测是否有物体落在网格中，并记录
        cell[idxH, idxW]
    3 计算包含物体中心点的cell的每个anchor的y_true
        每个cell包含3个anchor
        每个anchor包含数据：(3, num_gt, 2 + 5 + B*5)
    4 每张图片的数据写入json。（后面当y_true使用）
        一张图片一个json，包含：
            {
                img_fname: 图片名称(不含png)
                scales: [
                    {
                        scale: int               特征图缩放比例
                        fmaps_scale: [H1, W1]    特征图高度，宽度（该尺寸最终会将原图缩放成[H1 * W1]）
                        cells: [             负责检测物体的cell信息
                            {
                                idxH:            相对特征图的h轴坐标
                                idxW:            相对特征图的w轴坐标
                                gt: {            cell负责检测的物体信息
                                    x: cell负责物体的中心点x坐标相对cell左上点x坐标的偏移（相对特征图）
                                    y: cell负责物体的中心点y坐标相对cell左上点y坐标的偏移（相对特征图）
                                    w: cell负责物体的宽度与整图宽度占比
                                    h: cell负责物体的高度与整图高度占比
                                    idxV: cell负责物体的实际分类索引
                                }
                                anchors: [    cell中的anchor信息
                                    {
                                        iou:     anchor相对于gt的IoU
                                        w:       anchor的宽度于整图占比
                                        h:       anchor的高度于整图占比
                                    }
                                    ...
                                ]
                            }
                        ]
                    }
                    ...
                ]
            }

@author: luoyi
Created on 2021年2月22日
'''
import numpy as np
import os
import json
import tensorflow as tf

import utils.conf as conf
import utils.alphabet as alphabet
import utils.logger_factory as logf
from utils.iou import iou_n21_np, iou_121_np
from math import floor
from data.dataset import get_fpaths, read_image_to_arr


log = logf.get_logger('data_cells')


#    Cell生成器
class CellCreator():
    def __init__(self,
                 standard_scale=[conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT],
                 scales_set=conf.DATASET_CELLS.get_scales_set(),
                 anchors_set=conf.DATASET_CELLS.get_anchors_set()):
        '''
            @param standard_scale: 图片统一化缩放尺寸
            @param scales: 特征图缩放比例集合
            @param anchors: numpy(num_scale, num_anchor, 2)
                                num_scale: 一共有多少个尺寸
                                num_anchor: 每个尺寸下对应多少个anchor框
                                2(W,H): 每个anchor框的W, H
        '''
        #    检测scales.shape[0]与anchors.shape[0]是否一致
        assert (scales_set.shape[0] == anchors_set.shape[0]), 'scales.shape[0] must equal anchors.shape[0] scales.shape:' + str(scales_set.shape) + ' anchors.shape:' + str(anchors_set.shape)
        
        self._standard_scale = standard_scale
        self._scales_set = scales_set
        self._anchors_set = anchors_set
        
        #    根据缩放比例计算每个比例下的特征图尺寸
        fmaps_scales = np.repeat(np.expand_dims(standard_scale, axis=0), repeats=scales_set.shape[0], axis=0) / np.expand_dims(scales_set, axis=-1)
        self._fmaps_scales = np.ceil(fmaps_scales).astype(np.int32)
        pass
    
    #    迭代所有样本，并写入json
    def create(self,
               data_iterator=None,
               count=-1,
               out_path='',
               show_log=True,
               log_interval=100):
        '''
            @param data_iterator: 数据迭代器
            @param count: 总共迭代多少个数据。小于0表示直到data_iteratord读完为止
            @param out_path: json文件往哪写
            @param show_log: 生成过程中是否打日志
            @param log_interval: 每隔多少条记录打一次日志
        '''
        conf.mkfiledir_ifnot_exises(out_path)
        if (os.path.exists(out_path)): os.remove(out_path)
        fw = open(file=out_path, mode='w', encoding='utf-8')
        
        crt_count = 0
        #    迭代所有样本
        for fname, img_shape, _, _, labels in data_iterator:
            crt_count += 1
            if (count > 0 and crt_count > count): break
            #    预处理标签信息
            label = self._preprocess_label(labels=labels, img_shape=img_shape)
            #    scales_set尺度下，负责预测的cell信息
            cells_info = self._split_cell_anchor(img_scale=self._standard_scale, 
                                                 labels=label, 
                                                 scales_set=self._scales_set, 
                                                 fmaps_scales=self._fmaps_scales,
                                                 anchors_set=self._anchors_set)
            
            d = {'img_fname':fname, 'scales':cells_info}
            j = json.dumps(d)
            fw.write(j + '\n')
            
            if (show_log and crt_count % log_interval == 0): 
                log.info('create cells. crt_count:' + str(crt_count) + ' count:' + str(count))
                pass
            pass
        
        fw.close()
        pass
    
    #    单一样本生成
    def create_one_sample(self,
                          img=None,
                          labels=None):
        '''
            @param img: 图片数组 numpy(H, W, 3)
            @param labels: 标签数据 list [v, x,y, w,h]相对原图, (x,y左上点坐标)
            @return: list [
                            {
                                scale: int               特征图缩放比例
                                fmaps_scale: [H1, W1]    特征图高度，宽度（该尺寸最终会将原图缩放成[H1 * W1]）
                                cells: [                 负责检测物体的cell信息
                                    {
                                        idxH:            相对特征图的h轴坐标
                                        idxW:            相对特征图的w轴坐标
                                        gt: {            cell负责检测的物体信息
                                            x: cell负责物体的中心点x坐标相对cell左上点x坐标的偏移（相对特征图）
                                            y: cell负责物体的中心点y坐标相对cell左上点y坐标的偏移（相对特征图）
                                            w: cell负责物体的宽度与整图宽度占比
                                            h: cell负责物体的高度与整图高度占比
                                            idxV: cell负责物体的实际分类索引
                                        }
                                        anchors: [    cell中的anchor信息
                                            {
                                                iou:     anchor相对于gt的IoU
                                                w:       anchor的宽度于整图占比
                                                h:       anchor的高度于整图占比
                                            }
                                            ...
                                        ]
                                    }
                                ]
                            },
                            ...
                        ]
        '''
        label = self._preprocess_label(labels=labels, img_shape=img.shape)
        cells_info = self._split_cell_anchor(img_scale=self._standard_scale, 
                                             labels=label, 
                                             scales_set=self._scales_set, 
                                             fmaps_scales=self._fmaps_scales,
                                             anchors_set=self._anchors_set)
        return cells_info
    
    #    预处理label信息
    def _preprocess_label(self, labels=None, img_shape=None):
        '''
            @param labels: 标签信息 [v, x,y(左上点), w,h] 相对原始尺寸原图
            @param img_shape: 图像原始尺寸
            @return: [idxV, xl,yl, xr,yr] 相对统一尺寸原图
        '''
        #    只保留x,y,w,h信息。并且按照统一比例缩放
        scale_h = img_shape[0] / self._standard_scale[0]
        scale_w = img_shape[1] / self._standard_scale[1]
        #    取所有标签信息
        ls = []
        for label in labels:
            #    取左上点坐标
            lx = label[1]
            ly = label[2]
            #    取宽高，并换算成右下点坐标
            rx = lx + label[3]
            ry = ly + label[4]
            #    按比例缩放
            lx = lx / scale_w
            ly = ly / scale_h
            rx = rx / scale_w
            ry = ry / scale_h
            #    分类索引
            v = alphabet.category_index(label[0])
            #    记录box
            ls.append([lx,ly, rx,ry, v])
            pass
        #    标签信息numpy(num, 5)
        ls = np.array(ls)
        return ls
    
    #    按照比例划分cell，并计算负责预测物体的cell，并计算cell中的anchor
    def _split_cell_anchor(self,
                           img_scale=[conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT],
                           labels=None,
                           scales_set=conf.DATASET_CELLS.get_scales_set(),
                           anchors_set=conf.DATASET_CELLS.get_anchors_set(),
                           fmaps_scales=None):
        '''
            @param img_scale: 图片尺寸
            @param labels: 标签信息numpy(num_label, 5)
                                [xl,yl, xr,yr, vidx]
            @param scales_set: 特整图缩放比例集合[8, 16, 32]
            @param fmaps_scales: 根据缩放比例和统一规格尺寸换算出的特征图尺寸 numpy(num_scale, 2)
            @param anchors_set: 每个cell内的anchor numpy(num_scales, num_anchors, 2)
            @return: list [
                            {
                                scale: int               特征图缩放比例
                                fmaps_scale: [H1, W1]    特征图高度，宽度（该尺寸最终会将原图缩放成[H1 * W1]）
                                cells: [             负责检测物体的cell信息
                                    {
                                        idxH:            相对特征图的h轴坐标
                                        idxW:            相对特征图的w轴坐标
                                        gt: {            cell负责检测的物体信息
                                            x: cell负责物体的中心点x坐标相对cell左上点x坐标的偏移（相对特征图）
                                            y: cell负责物体的中心点y坐标相对cell左上点y坐标的偏移（相对特征图）
                                            w: cell负责物体的宽度与整图宽度占比
                                            h: cell负责物体的高度与整图高度占比
                                            idxV: cell负责物体的实际分类索引
                                        }
                                        anchors: [    cell中的anchor信息
                                            {
                                                iou:     anchor相对于gt的IoU
                                                w:       anchor的宽度于整图占比
                                                h:       anchor的高度于整图占比
                                            }
                                            ...
                                        ]
                                    }
                                ]
                            },
                            ...
                        ]
        '''
        res = []
        #    循环每一个scales
        for i in range(len(scales_set)):
            scale = scales_set[i]
            anchors = anchors_set[i]
            fmaps_scale = fmaps_scales[i]
            
            scale_info = self._calculation_cell_label_anchor_info(scale=scale,
                                                                  fmaps_scale=fmaps_scale,
                                                                  anchors=anchors,
                                                                  labels=labels,
                                                                  img_scale=img_scale)
            res.append(scale_info)
            pass
        return res
    
    #    计算负责检测的cell，和其对应的label，和其对应的anchor信息
    def _calculation_cell_label_anchor_info(self, 
                                            scale=None, 
                                            fmaps_scale=None,
                                            anchors=None, 
                                            labels=None, 
                                            img_scale=[conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT]):
        ''' 计算负责检测的cell，和其对应的label，和其对应的anchor信息
            @param img_scale: 图片尺寸
            @param labels: 标签信息numpy(num_label, 5)
                                [xl,yl, xr,yr, vidx]
            @param scale: 特征图缩放比例 int (8 | 16 | 32)
            @param fmaps_scale: 根据特征图缩放比例和统一图片尺寸换算出的特征图尺寸 numpy(2,)
            @param anchors: scale对应的anchors numpy(1, num_anchor, 2)
            @return dict  
                            {
                                scale: int               特征图缩放比例
                                fmaps_scale: [H1, W1]    特征图高度，宽度（该尺寸最终会将原图缩放成[H1 * W1]）
                                cells: [             负责检测物体的cell信息
                                    {
                                        idxH:            相对特征图的h轴坐标
                                        idxW:            相对特征图的w轴坐标
                                        gt: {            cell负责检测的物体信息
                                            x: cell负责物体的中心点x坐标相对cell左上点x坐标的偏移（相对特征图）
                                            y: cell负责物体的中心点y坐标相对cell左上点y坐标的偏移（相对特征图）
                                            w: cell负责物体的宽度与整图宽度占比
                                            h: cell负责物体的高度与整图高度占比
                                            idxV: cell负责物体的实际分类索引
                                        }
                                        anchors: [    cell中的anchor信息
                                            {
                                                iou:     anchor相对于gt的IoU
                                                w:       anchor的宽度于整图占比
                                                h:       anchor的高度于整图占比
                                            }
                                            ...
                                        ]
                                    }
                                    ...
                                ]
                            }
        '''
        #    计算每个label相对特征图的中心点，并按scale缩放到指定倍数。（换算到特征图上的坐标）
        lxc = (labels[:,0] + (labels[:,2] - labels[:,0]) / 2.) / scale
        lyc = (labels[:,1] + (labels[:,3] - labels[:,1]) / 2.) / scale
        #    计算每个label相对特征图的宽高（换算到特征图上的宽高）
        lw = (labels[:,2] - labels[:,0]) / scale
        lh = (labels[:,3] - labels[:,1]) / scale
        #    [xc,yc, w,h, idxV]
        linfo = np.concatenate([np.expand_dims(lxc, axis=-1), 
                                np.expand_dims(lyc, axis=-1),
                                np.expand_dims(lw, axis=-1),
                                np.expand_dims(lh, axis=-1),
                                np.expand_dims(labels[:,4], axis=-1)], axis=-1)
        
        #    anchor的宽高按比例缩放到特征图尺寸（换算到特征图上的宽高）
        fmaps_anchors_h = anchors[:,0] / scale
        fmaps_anchors_w = anchors[:,1] / scale
        fmaps_anchors = np.concatenate([np.expand_dims(fmaps_anchors_h, axis=-1),
                                        np.expand_dims(fmaps_anchors_w, axis=-1)], axis=-1)
        
        #    定义返回值（nparray对象貌似不能被序列化）
        res = {'scale':int(scale), 'fmaps_scale':fmaps_scale.tolist()}
        
        cells_map = {}
        #    lxc和lyc下取整就是包含label的cell相对特征图的坐标
        #    逐个cell追加idxH, idxW, gt，anchors信息
        #    可能会存在多个物体落在同一个cell中的情况。多个物体时取cell与gt的IoU大的那个物体
        for info in linfo:
            idxW = floor(info[0])
            idxH = floor(info[1])
            if (idxH >= fmaps_scale[0] or idxW >= fmaps_scale[1]):
                print(idxH, idxW, fmaps_scale)
            
            k = (idxH, idxW)
            #    判断(idxH, idxW)之前有没有物体落入
            cell = cells_map.get(k)
            #    若(idxH, idxW)之前有物体，则判断两个物体与cell的IoU，取较大的那个
            if (cell is not None):
                box_prev = [idxW + cell['gt']['x'] - cell['gt']['w']/2, 
                            idxH + cell['gt']['y'] - cell['gt']['h']/2,
                            idxW + cell['gt']['x'] + cell['gt']['w']/2,
                            idxH + cell['gt']['y'] + cell['gt']['h']]
                
                box_cell = [idxW + 0.5 - fmaps_scale[1]/2,
                            idxH + 0.5 - fmaps_scale[0]/2,
                            idxW + 0.5 + fmaps_scale[1]/2,
                            idxH + 0.5 + fmaps_scale[0]/2,]
                
                iou_prev = iou_121_np(box_cell, box_prev)
                iou_crt = iou_121_np(box_cell, info)
                
                #    如果之前的iou比现在的大，则跳过现在的gt
                if (iou_prev >= iou_crt): continue
                pass
            
            #    记录idxH, idxW信息
            cell = {'idxH': idxH, 'idxW': idxW}
            #    记录gt信息
            cell['gt'] = {'x': info[0] - idxW,
                          'y': info[1] - idxH,
                          'w': info[2] / fmaps_scale[1],
                          'h': info[3] / fmaps_scale[0],
                          'idxV': info[4]}
            #    记录anchor信息
            anchors_info = self._calculation_anchors_info(idxCell=(idxH, idxW), 
                                                          scale=scale, 
                                                          fmaps_scale=fmaps_scale, 
                                                          anchors=fmaps_anchors, 
                                                          linfo=info)
            anchors_list = []
            for anchor in anchors_info:
                anchors_list.append({'iou':anchor[0], 'w':anchor[1], 'h':anchor[2]})
                pass
            cell['anchors'] = anchors_list
            
            #    记录cell
            cells_map[(idxH, idxW)] = cell
            pass
        res['cells'] = list(cells_map.values())

        return res
    
    #    检测第[h,w]的cell中包含物体信息
    def _calculation_cell_contain_label_info(self, idxCell=None, linfo=None):
        '''
            @param idxCell: cell的索引(idxH, idxW)
            @param linfo: label信息 numpy(num_label, 5)
                            [label中心点相对特征图坐标，label宽高相对整图占比，label分类索引]
                            [xc,yc, w,h, idxV]
            @return: liable_cells cell中包含的物体 numpy(num, 5)
                                    num > 0即为检测出物体
                                    [xc,yc, w,h, idxV]
        '''
        #    如果某一个label的(xc,yc) - cell的(idxW, idxH)，结果都在(0~1)之间，则判定该cell负责检测该物体
        difference_x = linfo[:,0] - idxCell[1]
        difference_y = linfo[:,1] - idxCell[0]
        difference = np.concatenate([np.expand_dims(difference_x, axis=-1),
                                     np.expand_dims(difference_y, axis=-1)], axis=-1)
        mind, maxd = 0, 1
        liable_cells = linfo[((difference[:,0] > mind) * (difference[:,0] < maxd)) * ((difference[:,1] > mind) * (difference[:,1] < maxd))]
        return liable_cells
    
    #    计算anchors信息
    def _calculation_anchors_info(self, idxCell=None, scale=None, fmaps_scale=None, anchors=None, linfo=None):
        '''计算IoU(anchor, GT), scale_w, scale_h, 
            @param idxCell: cell的索引(idxH, idxW)
            @param scale: 特征图缩放比例 int (8, 6, 32)
            @param fmaps_scale: 根据特征图缩放比例和原图统一尺寸换算出的特征图尺寸 numpy(2,)
            @param anchors: anchor信息 numpy(num_anchor, 2(h,w)) 相对特征图的宽高
            @param linfo: cell中的物体信息 numpy(1, 5)
                                            [xc,yc, w,h, idxV]
            @return: numpy(num_anchor, 3)
                        [IoU, w, h]
        '''
        #    计算每个anchor与label的IoU
        #    根据idxH,idxW, anchor的宽高还原anchor相对特征图的xl,yl, xr,yr
        anchors_xc = np.repeat(idxCell[1] + 0.5, repeats=anchors.shape[0])
        anchors_yc = np.repeat(idxCell[0] + 0.5, repeats=anchors.shape[0])
        anchors_xl, anchors_xr = anchors_xc - anchors[:,1]/2, anchors_xc + anchors[:,1]/2
        anchors_yl, anchors_yr = anchors_yc - anchors[:,0]/2, anchors_yc + anchors[:,0]/2
        rect_srcs = np.concatenate([np.expand_dims(anchors_xl, axis=-1),
                                    np.expand_dims(anchors_yl, axis=-1),
                                    np.expand_dims(anchors_xr, axis=-1),
                                    np.expand_dims(anchors_yr, axis=-1)], axis=-1)
        #    根据linfo还原label相对特征图的xl,yl, xr,yr
        rect_tag = np.array([linfo[0] - linfo[2]/2,
                             linfo[1] - linfo[3]/2,
                             linfo[0] + linfo[2]/2,
                             linfo[1] + linfo[2]/2])
        #    计算每个anchor与label的IoU（用相对特征图的坐标计算）
        anchors_iou = iou_n21_np(rect_srcs, rect_tag)
        #    anchor的宽高占比
        anchors_h = anchors[:,0] / fmaps_scale[0]
        anchors_w = anchors[:,1] / fmaps_scale[1]
        return np.concatenate([np.expand_dims(anchors_iou, axis=-1),
                               np.expand_dims(anchors_w, axis=-1),
                               np.expand_dims(anchors_h, axis=-1)], axis=-1)
    pass


#    cell空数据（补全长度用，不然y做不成一个tensor）
cell_empty = [-1,-1, -1,-1,-1,-1,-1]
for _ in range(conf.DATASET_CELLS.get_anchors_set().shape[1]): cell_empty += [-1, -1, -1]

#    cells 标签数据迭代器
def cells_iterator(img_dir=conf.DATASET_CELLS.get_in_train(),
                   label_path=conf.DATASET_CELLS.get_label_train(),
                   is_label_mutiple=conf.DATASET_CELLS.get_label_train_mutiple(),
                   count=conf.DATASET_CELLS.get_count_train(),
                   x_preprocess=lambda x:((x / 255.) - 0.5) * 2,                #    默认图片矩阵归一化到[-1,1]之间
                   y_preprocess=None,
                   ):
    #    取所有的标签文件名
    label_files = get_fpaths(is_mutiple_file=is_label_mutiple, file_path=label_path)
    
    for label_file in label_files:
        crt_count = 0
        #    每个文件取count个样本，如果文件中数量不够，则循环读取直到够数为止
        while (crt_count < count):
            for line in open(file=label_file, mode='r', encoding='utf-8'):
                crt_count += 1
                if (crt_count > count): break
                
                d = json.loads(line)
                
                #    读图片数据，并做成x
                img_file = d['img_fname']
                x, _ = read_image_to_arr(img_path=img_dir + '/' + img_file + '.png')
                if (x_preprocess): x = x_preprocess(x)
                
                #    读scales数据，并做成y
                scales = d['scales']
                y = []
                for scale_data in scales:
#                     scale = scale_data['scale']
#                     fmaps_scale = scale_data['fmaps_scale']
                    cells = scale_data['cells']
                    y_scale = []
                    for cell_data in cells:
                        #    cell数据格式：[idxH, idxW, gt_x, gt_y, gt_w, gt_h, gt_idxV, B*(anchor_iou, anchor_w, anchor_h)]
                        #    cell坐标
                        idxH = cell_data['idxH']
                        idxW = cell_data['idxW']
                        #    gt信息
                        gt_x = cell_data['gt']['x']
                        gt_y = cell_data['gt']['y']
                        gt_w = cell_data['gt']['w']
                        gt_h = cell_data['gt']['h']
                        gt_idxV = cell_data['gt']['idxV']
                        #    anchor信息
                        anchors = []
                        for anchor_data in cell_data['anchors']:
                            anchors += [anchor_data['iou'], anchor_data['w'], anchor_data['h']]
                            pass
                        cell = [idxH, idxW, gt_x, gt_y, gt_w, gt_h, gt_idxV] + anchors
                        y_scale.append(cell)
                        pass
                    #    如果y_scale的长度不足6，则补全-1数据
                    if (len(y_scale) < 6): 
                        for _ in range(6 - len(y_scale)):
                            y_scale.append(cell_empty)
                            pass
                        pass
                    y.append(y_scale)
                    pass
                y = np.array(y, dtype=np.float64)
                if (y_preprocess): y = y_preprocess(y)
                
                yield x, y
                pass
            
            #    如果读完文件后crt_count还是0，则判定文件为空
            assert (crt_count > 0), 'label file is empty. label_file:' + label_file
            pass
        pass
    pass
#    cells数据集
def tensor_db(img_dir=conf.DATASET_CELLS.get_in_train(),
              label_path=conf.DATASET_CELLS.get_label_train(),
              is_label_mutiple=conf.DATASET_CELLS.get_label_train_mutiple(),
              count=conf.DATASET_CELLS.get_count_train(),
              batch_size=conf.DATASET_CELLS.get_batch_size(),
              epochs=conf.DATASET_CELLS.get_epochs(),
              shuffle_buffer_rate=conf.DATASET_CELLS.get_shuffle_buffer_rate(),
              x_preprocess=lambda x:((x / 255.) - 0.5) * 2,
              y_preprocess=None):
    x_shape = tf.TensorShape([conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT, 3])
    y_shape = tf.TensorShape([len(conf.DATASET_CELLS.get_scales_set()), 6, 2 + 5 + conf.DATASET_CELLS.get_anchors_set().shape[1] * 3])
    db = tf.data.Dataset.from_generator(generator=lambda :cells_iterator(img_dir=img_dir,
                                                                         label_path=label_path,
                                                                         is_label_mutiple=is_label_mutiple,
                                                                         count=count,
                                                                         x_preprocess=x_preprocess,
                                                                         y_preprocess=y_preprocess), 
                                        output_types=(tf.float32, tf.float64), 
                                        output_shapes=(x_shape, y_shape))
    if (shuffle_buffer_rate > 0): db = db.shuffle(buffer_size=shuffle_buffer_rate * batch_size)
    if (batch_size): db = db.batch(batch_size)
    if (epochs): db = db.repeat(epochs)
    return db


