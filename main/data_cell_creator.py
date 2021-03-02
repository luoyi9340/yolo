# -*- coding: utf-8 -*-  
'''
包含物体的cell数据集

@author: luoyi
Created on 2021年2月24日
'''
import sys
import os
#    取项目根目录
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('yolo')[0]
ROOT_PATH = ROOT_PATH + "yolo"
sys.path.append(ROOT_PATH)

import utils.conf as conf
import utils.logger_factory as logf
import data.dataset_cells as ds_cells
import data.dataset as ds


log = logf.get_logger('data_cells')


log.info('create cells dataset...')
#    cells数据生成器
cell_creator = ds_cells.CellCreator(standard_scale=[conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT],
                                    scales_set=conf.DATASET_CELLS.get_scales_set(),
                                    anchors_set=conf.DATASET_CELLS.get_anchors_set())
log.info('init cell_creator standard_scale:' + str([conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT]) + 'scales_set:' + str(conf.DATASET_CELLS.get_scales_set()))

#    训练数据集
train_db_iter = ds.data_iterator(img_dir=conf.DATASET.get_in_train(), 
                                 label_path=conf.DATASET.get_label_train(), 
                                 is_mutiple_file=conf.DATASET.get_label_train_mutiple(), 
                                 count=conf.DATASET.get_count_train())
log.info('init train_db_iter. count:' + str(conf.DATASET.get_count_train()) + 'img_dir:' + str(conf.DATASET.get_in_train()) + ' label_path:' + str(conf.DATASET.get_label_train()))
#    验证数据集
val_db_iter = ds.data_iterator(img_dir=conf.DATASET.get_in_val(), 
                               label_path=conf.DATASET.get_label_val(), 
                               is_mutiple_file=conf.DATASET.get_label_val_mutiple(), 
                               count=conf.DATASET.get_count_val())
log.info('init val_db_iter. count:' + str(conf.DATASET.get_count_val()) + 'img_dir:' + str(conf.DATASET.get_in_val()) + ' label_path:' + str(conf.DATASET.get_label_val()))
#    测试数据集
test_db_iter = ds.data_iterator(img_dir=conf.DATASET.get_in_test(), 
                                label_path=conf.DATASET.get_label_test(), 
                                is_mutiple_file=conf.DATASET.get_label_test_mutiple(), 
                                count=conf.DATASET.get_count_test())
log.info('init test_db_iter. count:' + str(conf.DATASET.get_count_test()) + 'img_dir:' + str(conf.DATASET.get_in_test()) + ' label_path:' + str(conf.DATASET.get_label_test()))

#    生成训练数据
cell_creator.create(data_iterator=train_db_iter, 
                    count=conf.DATASET_CELLS.get_count_train(), 
                    out_path=conf.DATASET_CELLS.get_label_train())
#    生成验证数据集
cell_creator.create(data_iterator=val_db_iter, 
                    count=conf.DATASET_CELLS.get_count_val(), 
                    out_path=conf.DATASET_CELLS.get_label_val())
#    生成测试数据集
cell_creator.create(data_iterator=train_db_iter, 
                    count=conf.DATASET_CELLS.get_count_test(), 
                    out_path=conf.DATASET_CELLS.get_label_test())


