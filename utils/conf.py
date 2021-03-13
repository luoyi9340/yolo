# -*- coding: utf-8 -*-  
'''
日志组件

@author: luoyi
Created on 2021年3月2日
'''
import yaml
import os
import sys
import numpy as np


#    取项目根目录（其他一切相对目录在此基础上拼接）
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('yolo')[0]
ROOT_PATH = ROOT_PATH + "yolo"

#    训练图片统一高度
IMAGE_HEIGHT = 180
#    训练图片基本宽度（根据vcode长度有所不同，每个字120。自己算吧）
IMAGE_BASE_WEIGHT = 120
IMAGE_WEIGHT = 4 * IMAGE_BASE_WEIGHT

#    最大验证码数
MAX_VCODE_NUM = 6


#    取配置文件目录
CONF_PATH = ROOT_PATH + "/resources/conf.yml"
#    加载conf.yml配置文件
def load_conf_yaml(yaml_path=CONF_PATH):
    print('加载配置文件:' + CONF_PATH)
    f = open(yaml_path, 'r', encoding='utf-8')
    fr = f.read()
    
#     c = yaml.load(fr, Loader=yaml.SafeLoader)
    c = yaml.safe_load(fr)
    
    #    读取letter相关配置项
    dataset = Dataset(c['dataset']['in_train'], c['dataset']['count_train'], c['dataset']['label_train'], c['dataset']['label_train_mutiple'],
                      c['dataset']['in_val'], c['dataset']['count_val'], c['dataset']['label_val'], c['dataset']['label_val_mutiple'],
                      c['dataset']['in_test'], c['dataset']['count_test'], c['dataset']['label_test'], c['dataset']['label_test_mutiple'],
                      c['dataset']['batch_size'],
                      c['dataset']['epochs'],
                      c['dataset']['shuffle_buffer_rate'])
    
    dataset_cells = DatasetCells(c['dataset_cells']['anchors_set'],
                                 c['dataset_cells']['scales_set'],
                                 c['dataset_cells']['in_train'], c['dataset_cells']['count_train'], c['dataset_cells']['label_train'], c['dataset_cells']['label_train_mutiple'],
                                 c['dataset_cells']['in_val'], c['dataset_cells']['count_val'], c['dataset_cells']['label_val'], c['dataset_cells']['label_val_mutiple'],
                                 c['dataset_cells']['in_test'], c['dataset_cells']['count_test'], c['dataset_cells']['label_test'], c['dataset_cells']['label_test_mutiple'],
                                 c['dataset_cells']['batch_size'],
                                 c['dataset_cells']['epochs'],
                                 c['dataset_cells']['shuffle_buffer_rate'])
    
    v4 = V4(c['v4']['learning_rate'],
            c['v4']['save_weights_dir'],
            c['v4']['tensorboard_dir'],
            c['v4']['loss_lamda_box'],
            c['v4']['loss_lamda_confidence'],
            c['v4']['loss_lamda_unconfidence'],
            c['v4']['loss_lamda_cls'],
            c['v4']['threshold_liable_iou'],
            c['v4']['max_objects'],
            c['v4']['threshold_overlap_iou'])
    return c, dataset, dataset_cells, v4


#    验证码识别数据集。为了与Java的风格保持一致
class Dataset:
    def __init__(self, 
                 in_train="", count_train=50000, label_train="", label_train_mutiple=False,
                 in_val="", count_val=10000, label_val="", label_val_mutiple=False,
                 in_test="", count_test=10000, label_test="", label_test_mutiple=False,
                 batch_size=2, epochs=2, shuffle_buffer_rate=-1):
        self.__in_train = convert_to_abspath(in_train)
        self.__count_train = count_train
        self.__label_train = convert_to_abspath(label_train)
        self.__label_train_mutiple = label_train_mutiple
        
        self.__in_val = convert_to_abspath(in_val)
        self.__count_val = count_val
        self.__label_val = convert_to_abspath(label_val)
        self.__label_val_mutiple = label_val_mutiple
        
        self.__in_test = convert_to_abspath(in_test)
        self.__count_test = count_test
        self.__label_test = convert_to_abspath(label_test)
        self.__label_test_mutiple = label_test_mutiple
        
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__shuffle_buffer_rate = shuffle_buffer_rate
        pass
    def get_in_train(self): return convert_to_abspath(self.__in_train)
    def get_count_train(self): return self.__count_train
    def get_label_train(self): return convert_to_abspath(self.__label_train)
    def get_label_train_mutiple(self): return self.__label_train_mutiple
    
    def get_in_val(self): return convert_to_abspath(self.__in_val)
    def get_count_val(self): return self.__count_val
    def get_label_val(self): return convert_to_abspath(self.__label_val)
    def get_label_val_mutiple(self): return self.__label_val_mutiple   
    
    def get_in_test(self): return convert_to_abspath(self.__in_test)
    def get_count_test(self): return self.__count_test
    def get_label_test(self): return convert_to_abspath(self.__label_test)
    def get_label_test_mutiple(self): return self.__label_test_mutiple
    
    def get_batch_size(self): return self.__batch_size
    def get_epochs(self): return self.__epochs
    def get_shuffle_buffer_rate(self): return self.__shuffle_buffer_rate
    pass


class DatasetCells:
    def __init__(self,
                 anchors_set=[[[81.54752427957331, 56.43448495462505], 
                               [81.10758580905683, 59.1673877511779], 
                               [81.3733002779656, 59.538827536122994]], 
                              [[81.15771939123192, 63.73158173695768], 
                               [80.48762283866152, 64.94895716714359], 
                               [81.273, 66.70218333333332]], 
                              [[81.86197916666667, 68.10555555555555], 
                               [81.28434504792332, 74.88210862619809], 
                               [82.79291824149593, 80.07459717525364]]],
                 scales_set=[[6, 15], 
                             [12, 30], 
                             [23, 60]],
                 in_train="", count_train=50000, label_train="", label_train_mutiple=False,
                 in_val="", count_val=10000, label_val="", label_val_mutiple=False,
                 in_test="", count_test=10000, label_test="", label_test_mutiple=False,
                 batch_size=2, epochs=2, shuffle_buffer_rate=-1
                 ):
        self.__anchors_set = np.array(anchors_set)
        self.__scales_set = np.array(scales_set)
        
        self.__in_train = convert_to_abspath(in_train)
        self.__count_train = count_train
        self.__label_train = convert_to_abspath(label_train)
        self.__label_train_mutiple = label_train_mutiple
        
        self.__in_val = convert_to_abspath(in_val)
        self.__count_val = count_val
        self.__label_val = convert_to_abspath(label_val)
        self.__label_val_mutiple = label_val_mutiple
        
        self.__in_test = convert_to_abspath(in_test)
        self.__count_test = count_test
        self.__label_test = convert_to_abspath(label_test)
        self.__label_test_mutiple = label_test_mutiple
        
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__shuffle_buffer_rate = shuffle_buffer_rate
        pass
    def get_anchors_set(self): return self.__anchors_set
    def get_scales_set(self): return self.__scales_set
    
    def get_in_train(self): return convert_to_abspath(self.__in_train)
    def get_count_train(self): return self.__count_train
    def get_label_train(self): return convert_to_abspath(self.__label_train)
    def get_label_train_mutiple(self): return self.__label_train_mutiple
    
    def get_in_val(self): return convert_to_abspath(self.__in_val)
    def get_count_val(self): return self.__count_val
    def get_label_val(self): return convert_to_abspath(self.__label_val)
    def get_label_val_mutiple(self): return self.__label_val_mutiple   
    
    def get_in_test(self): return convert_to_abspath(self.__in_test)
    def get_count_test(self): return self.__count_test
    def get_label_test(self): return convert_to_abspath(self.__label_test)
    def get_label_test_mutiple(self): return self.__label_test_mutiple
    
    def get_batch_size(self): return self.__batch_size
    def get_epochs(self): return self.__epochs
    def get_shuffle_buffer_rate(self): return self.__shuffle_buffer_rate
    pass


#    V4相关配置
class V4():
    def __init__(self,
                 learning_rate=0.001,
                 save_weights_dir='',
                 tensorboard_dir='',
                 loss_lamda_box=1.,
                 loss_lamda_confidence=1.,
                 loss_lamda_unconfidence=0.5,
                 loss_lamda_cls=1.,
                 threshold_liable_iou=0.5,
                 max_objects=6,
                 threshold_overlap_iou=0.5):
        self.__learning_rate = learning_rate
        self.__save_weights_dir = save_weights_dir
        self.__tensorboard_dir = tensorboard_dir
        self.__loss_lamda_box = loss_lamda_box
        self.__loss_lamda_confidence = loss_lamda_confidence
        self.__loss_lamda_unconfidence = loss_lamda_unconfidence
        self.__loss_lamda_cls = loss_lamda_cls
        self.__threshold_liable_iou = threshold_liable_iou
        self.__max_objects = max_objects
        self.__threshold_overlap_iou = threshold_overlap_iou
        pass
    def get_learning_rate(self): return self.__learning_rate
    def get_save_weights_dir(self): return convert_to_abspath(self.__save_weights_dir)
    def get_tensorboard_dir(self): return convert_to_abspath(self.__tensorboard_dir)
    
    def get_loss_lamda_box(self): return self.__loss_lamda_box
    def get_loss_lamda_confidence(self): return self.__loss_lamda_confidence
    def get_loss_lamda_unconfidence(self): return self.__loss_lamda_unconfidence
    def get_loss_lamda_cls(self): return self.__loss_lamda_cls
    
    def get_threshold_liable_iou(self): return self.__threshold_liable_iou
    def get_max_objects(self): return self.__max_objects
    def get_threshold_overlap_iou(self): return self.__threshold_overlap_iou
    pass


#    取配置的绝对目录
def convert_to_abspath(path):
    '''取配置的绝对目录
        "/"开头的目录原样输出
        非"/"开头的目录开头追加项目根目录
    '''
    if (path.startswith("/")):
        return path
    else:
        return ROOT_PATH + "/" + path
    
#    检测文件所在上级目录是否存在，不存在则创建
def mkfiledir_ifnot_exises(filepath):
    '''检测log所在上级目录是否存在，不存在则创建
        @param filepath: 文件目录
    '''
    _dir = os.path.dirname(filepath)
    if (not os.path.exists(_dir)):
        os.makedirs(_dir)
    pass
#    检测目录是否存在，不存在则创建
def mkdir_ifnot_exises(_dir):
    '''检测log所在上级目录是否存在，不存在则创建
        @param dir: 目录
    '''
    if (not os.path.exists(_dir)):
        os.makedirs(_dir)
    pass

ALL_DICT, DATASET, DATASET_CELLS, V4 = load_conf_yaml()


#    写入配置文件
def write_conf(_dict, file_path):
    '''写入当前配置项的配置文件
        @param dict: 要写入的配置项字典
        @param file_path: 文件path
    '''
    file_path = convert_to_abspath(file_path)
    mkfiledir_ifnot_exises(file_path)
    
    #    存在同名文件先删除
    if (os.path.exists(file_path)):
        os.remove(file_path)
        pass
    
    fw = open(file_path, mode='w', encoding='utf-8')
    yaml.safe_dump(_dict, fw)
    fw.close()
    pass


#    追加sys.path
def append_sys_path(path):
    path = convert_to_abspath(path)
    sys.path.append(path)
    print(sys.path)
    pass
