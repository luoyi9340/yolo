# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年2月24日
'''
import matplotlib.pyplot as plot

from data.dataset_cells import CellCreator
import utils.conf as conf
import data.dataset as ds


di = ds.data_iterator(img_dir=conf.DATASET.get_in_train(), 
                      label_path=conf.DATASET.get_label_train(), 
                      is_mutiple_file=conf.DATASET.get_label_train_mutiple(), 
                      count=conf.DATASET.get_count_train())
cell_creator = CellCreator(standard_scale=[conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT],
                           scales_set=conf.DATASET_CELLS.get_scales_set(),
                           anchors_set=conf.DATASET_CELLS.get_anchors_set())

idx = 10
i = 0
for img_fname, img_shape, img, vcode, labels in di:
    i += 1
    if (i > idx): break;
    pass
res = cell_creator.create_one_sample(img, labels)
print(res)
'''
[
                            {
                                scale: [H1, W1]      特征图高度，宽度（该尺寸最终会将原图缩放成[H1 * W1]）
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

#    在图上划出
fig = plot.figure()
ax = fig.add_subplot(1,1,1)

standard_scale=[conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT]
for r in res:
    fmaps_scale = r['fmaps_scale']
    scale = r['scale']
    
    cells = r['cells']
    for cell in cells:
        cell_lx = cell['idxW'] * scale
        cell_ly = cell['idxH'] * scale
        cell_w = scale
        cell_h = scale
        #    绘制检测的cell
        rect_cell = plot.Rectangle((cell_lx, cell_ly), cell_w, cell_h, fill=False, edgecolor='black', linewidth=1)
        ax.add_patch(rect_cell)
        
        #    绘制gt
        gt = cell['gt']
        gt_xc = (gt['x'] + cell['idxW']) * scale
        gt_yc = (gt['y'] + cell['idxH']) * scale
        gt_w = gt['w'] * fmaps_scale[1] * scale
        gt_h = gt['h'] * fmaps_scale[0] * scale
        rect_gt = plot.Rectangle((gt_xc - gt_w/2, gt_yc - gt_h/2), gt_w, gt_h, fill=False, edgecolor='blue', linewidth=1)
        ax.add_patch(rect_gt)
        
        #    绘制anchor
        anchors = cell['anchors']
        for anchor in anchors:
            anchor_xc = (cell['idxW'] + 0.5) * scale
            anchor_yc = (cell['idxH'] + 0.5) * scale
            anchor_w = anchor['w'] * fmaps_scale[1] * scale
            anchor_h = anchor['h'] * fmaps_scale[0] * scale
            rect_gt = plot.Rectangle((anchor_xc - anchor_w/2, anchor_yc - anchor_h/2), gt_w, gt_h, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect_gt)
            pass
        pass
    pass

#    绘制图像
X = img / 255.
plot.imshow(X)
plot.show()








