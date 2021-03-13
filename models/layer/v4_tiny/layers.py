# -*- coding: utf-8 -*-  
'''
yolo_v4-tiny


网络结构：
    输入：(180 * 480 * 3)
    ---------- layer 1 ----------
    Conv: [3 * 3 * 32] strides=2 padding=1 norm=BN active=LeakyReLU out=[90 * 240 * 32]
    Conv: [3 * 3 * 64] strides=2 padding=1 norm=BN active=LeakyReLU out=[45 * 120 * 64]
    ---------- layer 2 ----------
    CSPBlock: [filters=128] out=[23 * 60 * 128]
    ---------- layer 3 ----------
    CSPBlock: [filters=256] out=[12 * 30 * 256]
    记录：branch1 = [12 * 30 * 256]
    ---------- layer 4 ----------
    CSPBlock: [filters=512] out=[6 * 15 * 512]
    ---------- layer 5 ----------
    Conv: [3 * 3 * 512] strides=1 padding=1 norm=BN active=LeakyReLU out=[6 * 15 * 512]
    Conv: [1 * 1 * 256] strides=1 padding=0 norm=BN active=LeakyReLU out=[6 * 15 * 256]
    记录：branch2 = [6 * 15 * 256]
    ---------- yolohard 1 ----------
    Conv: [3 * 3 * 512] strides=1 padding=1 norm=BN active=LeakyReLU out=[6 * 15 * 512]
    Conv: [1 * 1 * num_anchors*num_classes+5] strides=1 padding=1 out=[6 * 15 * num_anchors*num_classes+5]
    Reshape: out=[6, 15, num_anchors, num_classes + 5]
    记录：yolohard1 = [6, 15, num_anchors, num_classes + 5]
    ---------- yolohard 2 ----------
    CSPUpSampleBlock: [filters=256] in=[branch1, branch2] out=[12 * 30 * 256]
    Conv: [1 * 1 * num_anchors*num_classes+5] strides=1 padding=0 out=[12 * 30 * num_anchors*num_classes+5]
    Reshape: out=[12, 30, num_anchors, num_classes + 5]
    记录：yolohard2 = [12, 30, num_anchors, num_classes + 5]
    
    2个输出即为YOLO V4-tiny的2个Scale输出
    (num_anchors*num_classes+5)定义：
        (num_anchors * num_classes)：对应的分类得分
        (num_anchors * 1)：预测置信度
        (num_anchors * 4)：预测的x,y相对cell左上角偏移量，w，h相对特征图宽高
        

@author: luoyi
Created on 2021年3月9日
'''
import tensorflow as tf

import utils.conf as conf
import utils.alphabet as alphabet
from models.layer.commons.part import Conv2DNormActive
from models.layer.commons.part import YoloHardRegisterLayer
from models.layer.v4_tiny.part import CSPBlockTinyLayer, CSPUpSampleBlockTinyLayer, YolohardActiveLayer


class YoloV4TinyLayer(tf.keras.layers.Layer):

    def __init__(self,
                 name='yolo_v4_tiny',
                 input_shape=(conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT, 3),
                 num_anchors=conf.DATASET_CELLS.get_anchors_set().shape[1],
                 num_classes=len(alphabet.ALPHABET),
                 out_yolohard1=None,
                 out_yolohard2=None,
                 **kwargs):
        super(YoloV4TinyLayer, self).__init__(name=name, **kwargs)

        #    +2是为了兼容手动padding（模型第一层就上ZeroPadding会存在load_weights的bug）
        self._input_shape = (input_shape[0] + 2, input_shape[1] + 2, input_shape[2])
        
        self._num_anchors = num_anchors
        self._num_classes = num_classes
        
        self._out_yolohard1 = out_yolohard1
        self._out_yolohard2 = out_yolohard2
        
        #    装配网络
        self._assembling()
        pass
    
    #    装配网络
    def _assembling(self):
        #    ---------- layer 1 ----------
        self._layer1 = tf.keras.models.Sequential([
                Conv2DNormActive(name='layer1_conv1',
                                 filters=32, 
                                 kernel_size=[3, 3], 
                                 strides=2, 
                                 padding='VALID', 
                                 active=tf.keras.layers.LeakyReLU(),
                                 input_shape=self._input_shape, output_shape=(90, 240, 32)),
                Conv2DNormActive(name='layer1_conv2',
                                 filters=64, 
                                 kernel_size=[3, 3], 
                                 strides=2, 
                                 padding=1, 
                                 active=tf.keras.layers.LeakyReLU(),
                                 input_shape=(90, 240, 32), output_shape=(45, 120, 64)),
            ], name='layer1')
        
        #    ---------- layer 2 ----------
        self._layer2 = CSPBlockTinyLayer(name='layer2',
                                         filters=128,
                                         pooling_padding=[1, 0],
                                         input_shape=(45, 120, 64), output_shape=(23, 60, 128))
        
        #    ---------- layer 3 ----------
        self._layer3 = CSPBlockTinyLayer(name='layer3',
                                         filters=256,
                                         pooling_padding=[1, 0],
                                         input_shape=(23, 60, 128), output_shape=(12, 30, 256))
        
        #    ---------- layer 4 ----------
        self._layer4 = CSPBlockTinyLayer(name='layer4',
                                         filters=512,
                                         input_shape=(12, 30, 256), output_shape=(6, 15, 512))
        
        #    ---------- layer 5 ----------
        self._layer5 = tf.keras.Sequential([
                Conv2DNormActive(name='layer5_conv1',
                                 filters=512, 
                                 kernel_size=[3, 3], 
                                 strides=1, 
                                 padding='SAME', 
                                 active=tf.keras.layers.LeakyReLU(),
                                 input_shape=(6, 15, 512), output_shape=(6, 15, 512)),
                Conv2DNormActive(name='layer5_conv2',
                                 filters=256, 
                                 kernel_size=[1, 1], 
                                 strides=1, 
                                 padding='SAME', 
                                 active=tf.keras.layers.LeakyReLU(),
                                 input_shape=(6, 15, 512), output_shape=(6, 15, 256))
            ], name='layer5')
        
        yolohard_filters = self._num_anchors * (self._num_classes + 5)
        #    ---------- yolohard 1 ----------
        yolohard1_shape = (6, 15, self._num_anchors, self._num_classes + 5)
        self._yolohard1 = tf.keras.Sequential([
                Conv2DNormActive(name='yolohard1_conv1',
                                 filters=512, 
                                 kernel_size=[3, 3], 
                                 strides=1, 
                                 padding='SAME', 
                                 active=tf.keras.layers.LeakyReLU(),
                                 input_shape=(6, 15, 256), output_shape=(6, 15, 512)),
                tf.keras.layers.Conv2D(name='yolohard1_conv2',
                                       filters=yolohard_filters,
                                       kernel_size=[1, 1],
                                       strides=1,
                                       padding='SAME',
                                       input_shape=(6, 15, 512)),
                tf.keras.layers.Reshape(target_shape=yolohard1_shape)
            ], name='yolohard1')
        
        #    ---------- yolohard 2 ----------
        yolohard2_shape = (12, 30, self._num_anchors, self._num_classes + 5)
        self._yolohard2_upsample = CSPUpSampleBlockTinyLayer(name='yolohard2_upsample',
                                                             filters=256,
                                                             input_shape1=(12, 30, 256),
                                                             input_shape2=(6, 15, 256),
                                                             output_shape=(12, 30, 256))
        self._yolohard2_conv = tf.keras.layers.Conv2D(name='yolohard2_conv2',
                                                      filters=yolohard_filters,
                                                      kernel_size=[1, 1],
                                                      strides=1,
                                                      padding='SAME',
                                                      input_shape=(12, 30, 256))
        self._yolohard2_reshape = tf.keras.layers.Reshape(target_shape=yolohard2_shape)
        
        #    yolohard激活层
        self._yolohard_active = YolohardActiveLayer()
        
        #    数据暂存
        self._yolohard_register = YoloHardRegisterLayer()
        pass
    
    def call(self, x, training=None, **kwargs):
        #    先做padding1（放模型第一层会存在load_weight的bug，所以只能放这）
        x = tf.pad(x, paddings=[[0,0], [1,1], [1,1], [0,0]])
        
        #    layer1
        y = self._layer1(x)
        
        #    layer2
        y = self._layer2(y)
        
        #    layer3
        y = self._layer3(y)
        branch1 = y                                 #    branch1 Tensor(batch_size, 12, 30, num_anchors * (num_classes + 5))
        
        #    layer4
        y = self._layer4(y)
        
        #    layer5
        y = self._layer5(y)
        branch2 = y                                 #    branch2 Tensor(batch_size, 6, 15, num_anchors * (num_classes + 5))
        
        #    yolohard1
        yolohard1 = self._yolohard1(y)              #    yolohard1 Tensor(batch_size, 6, 15, num_anchors, num_classes + 5)
        
        #    yolohard2
        y = self._yolohard2_upsample(x=y, x1=branch1, x2=branch2)
        y = self._yolohard2_conv(y)
        yolohard2 = self._yolohard2_reshape(y)      #    yolohard2 Tensor(batch_size, 12, 30, num_anchors, num_classes + 5)
        
        yolohard1 = self._yolohard_active(yolohard1)
        yolohard2 = self._yolohard_active(yolohard2)
        
        #    暂存yolohard数据
        self._yolohard_register(x=y, yolohard1=yolohard1, yolohard2=yolohard2)
        return y

    pass

