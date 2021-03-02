# -*- coding: utf-8 -*-  
'''
数据集相关

@author: luoyi
Created on 2021年2月22日
'''
import os
import json
import numpy as np
from PIL import Image


import utils.conf as conf


#    labels文件迭代器
def data_iterator(img_dir=conf.DATASET.get_in_train(),
                  label_path=conf.DATASET.get_label_train(),
                  is_mutiple_file=conf.DATASET.get_label_train_mutiple(),
                  count=conf.DATASET.get_count_train()):
    '''labels文件迭代器
        @param img_dir: 图片目录
        @param label_path: 标签文件路径
        @param is_mutiple_file: 文件是否为多文件
        @param count: 每个文件迭代多少样本（单文件不够数的循环读取，直到够数为止）
        
        @return img_fname, img_shape, img(H,W,C), vcode, labels[[vcode, x, y, w, h]](x,y相对于左上点坐标)
    '''
    lfiles = get_fpaths(is_mutiple_file=is_mutiple_file, file_path=label_path)
    
    for lfile in lfiles:
        #    针对单个文件当前已迭代数量
        crt_count = 0
        while crt_count < count:
            for line in open(file=lfile, mode='r', encoding='utf-8'):
                crt_count += 1
                if (crt_count > count): break
                
                j = json.loads(line)
                #    读取图片数据
                img_fname = j['fileName']
                img_path = img_dir + "/" + img_fname + '.png'
                img, img_shape = read_image_to_arr(img_path)
                
                #    读取标签数据
                vcode = j['vcode']
                annos = j['annos']
                labels = []
                #    统一缩放比例
                scale_h = img_shape[0] / conf.IMAGE_HEIGHT
                scale_w = img_shape[1] / conf.IMAGE_WEIGHT
                for anno in annos:
                    labels.append([anno['key'], 
                                   anno['x'] / scale_w, 
                                   anno['y'] / scale_h, 
                                   anno['w'] / scale_w, 
                                   anno['h'] / scale_h])
                    pass
                yield img_fname, img_shape, img, vcode, labels
                pass
            
            #    如果读过文件后crt_count数量还是0，则判定文件是空的
            if crt_count == 0:
                raise Exception('file is empty. lfile:' + lfile)
            pass
        pass
    pass


#    读取图片数据，并缩放至统一尺寸
def read_image_to_arr(img_path, unified_scale=(conf.IMAGE_WEIGHT, conf.IMAGE_HEIGHT)):
    image = Image.open(img_path, mode='r')
    img_shape = image.size                      #    (W, H)
    img_shape = (img_shape[1], img_shape[0])    #    顺序倒过来(H, W)
    image = image.resize(unified_scale, Image.ANTIALIAS)
    img = np.asarray(image, dtype=np.float32)
    return img, img_shape

#    根据is_mutiple_file和file_path取所有规则下的文件名
def get_fpaths(is_mutiple_file=False, file_path=None):
    '''根据is_mutiple_file和file_path取所有规则下的文件名
        单文件模式直接返回文件名
        多文件模式按照{file_path}0, {file_path}1这样的顺序往后取。直到后缀数字断了为止
        @param is_mutiple_file: 是否多文件模式
        @param file_path: 文件路径
    '''
    if (not is_mutiple_file): return [file_path]
    
    f_idx = 0
    fpaths = []
    while (os.path.exists(file_path + str(f_idx))):
        fpaths.append(file_path + str(f_idx))
        f_idx += 1
        pass
    return fpaths