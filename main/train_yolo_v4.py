# -*- coding: utf-8 -*-  
'''
训练yolo_v4版本

@author: luoyi
Created on 2021年3月2日
'''
import sys
import os
#    取项目根目录
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('yolo')[0]
ROOT_PATH = ROOT_PATH + "yolo"
sys.path.append(ROOT_PATH)

import utils.conf as conf
import utils.logger_factory as logf
import utils.alphabet as alphabet

import data.dataset_cells as ds_cells

import models.yolo as yolo

from models.layer.commons.part import YoloHardRegister
from models.layer.v4.preprocess import AnchorsRegister
from math import ceil


log = logf.get_logger('train_v4')


#    初始化数据集
#    训练集
count_train = conf.DATASET_CELLS.get_count_train()
batch_size = conf.DATASET_CELLS.get_batch_size()
epochs=conf.DATASET_CELLS.get_epochs()
db_train = ds_cells.tensor_db(img_dir=conf.DATASET_CELLS.get_in_train(),
                              label_path=conf.DATASET_CELLS.get_label_train(), 
                              is_label_mutiple=conf.DATASET_CELLS.get_label_train_mutiple(), 
                              count=count_train, 
                              batch_size=batch_size, 
                              epochs=epochs, 
                              shuffle_buffer_rate=conf.DATASET_CELLS.get_shuffle_buffer_rate(), 
                              x_preprocess=lambda x:((x - 255.) - 0.5) / 2, 
                              y_preprocess=None)
log.info('init dataset train... ')


#    验证集
db_val = ds_cells.tensor_db(img_dir=conf.DATASET_CELLS.get_in_val(),
                            label_path=conf.DATASET_CELLS.get_label_val(), 
                            is_label_mutiple=conf.DATASET_CELLS.get_label_val_mutiple(), 
                            count=conf.DATASET_CELLS.get_count_val(), 
                            batch_size=conf.DATASET_CELLS.get_batch_size(), 
                            epochs=conf.DATASET_CELLS.get_epochs(), 
                            shuffle_buffer_rate=conf.DATASET_CELLS.get_shuffle_buffer_rate(), 
                            x_preprocess=lambda x:((x - 255.) - 0.5) / 2, 
                            y_preprocess=None)
log.info('init dataset val... ')


#    初始化模型
yolo_v4  =yolo.YoloV4(learning_rate=conf.V4.get_learning_rate(), 
                      loss_lamda_box=conf.V4.get_loss_lamda_box(),
                      loss_lamda_confidence=conf.V4.get_loss_lamda_confidence(),
                      loss_lamda_unconfidence=conf.V4.get_loss_lamda_unconfidence(),
                      loss_lamda_cls=conf.V4.get_loss_lamda_cls(),
                      num_anchors=conf.DATASET_CELLS.get_anchors_set().shape[1],
                      num_classes=len(alphabet.ALPHABET),
                      yolohard_register=YoloHardRegister.instance(),
                      anchors_register=AnchorsRegister.instance(),)
log.info('init model yolo_v4...')
yolo_v4.show_info()


log.info('feed data...')
steps_per_epoch = ceil(count_train / batch_size)
#    喂数据
yolo_v4.train_tensor_db(db_train=db_train, 
                        db_val=db_val, 
                        steps_per_epoch=steps_per_epoch, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        auto_save_weights_after_traind=True, 
                        auto_save_weights_dir=conf.V4.get_save_weights_dir(), 
                        auto_learning_rate_schedule=True, 
                        auto_tensorboard=True, 
                        auto_tensorboard_dir=conf.V4.get_tensorboard_dir())

