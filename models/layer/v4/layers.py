# -*- coding: utf-8 -*-  
'''
YOLO V4模型
    
网络结构：
    输入：(180 * 480 * 3)
    ---------- layer 1 ----------
    Conv: [3 * 3 * 32] strides=1 padding=1 norm=BN active=Mish out=[180 * 480 * 32]
    ---------- layer 2 ----------
    CSPResBlock: filters=64 times=1 part=[1,1] out=[90 * 240 * 64]
    ---------- layer 3 ----------
    CSPResBlock: filters=128 times=2 part=[0.5, 0.5] out=[45 * 120 * 128]
    ---------- layer 4 ----------
    CSPResBlock: filters=256 times=8 part=[0.5, 0.5] out=[23 * 60 * 256]
    记录：branch1: out=[23 * 60 * 256]
    ---------- layer 5 ----------
    CSPResBlock: filters=512 times=8 part=[0.5, 0.5] out=[12 * 30 * 512]
    记录：branch2: out=[12 * 30 * 512]
    ---------- layer 6 ----------
    CSPResBlock: filters=1024 times=4 part=[0.5, 0.5] out=[6 * 15 * 1024]
    ---------- process1 ----------
    ConvBlock: [3 * 3 * [512,1024,512]] out=[6 * 15 * 512]
    SPP: [6*15, 4*10, 2*5, 1*1] out=[6 * 15 * 4096]
    ConvBlock: [3 * 3 * [512,1024,512]] out=[6 * 15 * 512]
    记录：y13: out=[6 * 15 * 512]
    ---------- process2 ----------
    输入1：
        Conv: [1*1*256] in=y13 strides=1 padding=0 norm=BN active=Mish out=[6 * 15 * 256]
        up_sample: out=[12 * 30 * 256]
    输入2：
        branch2: out=[12 * 30 * 512]
    Concat: [输入1，输入2] out=[12 * 30 * 768]
    ConvBlock: [3 * 3 * [256,512,256,512,256]] out=[12 * 30 * 256]
    记录：y26: out=[12 * 30 * 256]
    ---------- process3 ----------
    输入1：
        Conv: [1*1*128] in=y26 strides=1 padding=0 norm=BN active=Mish out=[12 * 30 * 128]
        up_sample: out=[23 * 60 * 128]
    输入2：
        branch1: out=[23 * 60 * 256]
    Concat: [输入1，输入2] out=[23 * 60 * 384]
    ConvBlock: [3 * 3 * [128,256,128,256,128]] out=[23 * 60 * 128]
    ---------- YOLO HEAD1 ----------
    Conv: [3*3*256] in=process3 strides=1 padding=1 norm=BN active=Mish out=[23 * 60 * 256]
    Conv: [1*1*(num_anchors * (num_classes + 1))] strides=1 padding=0 out=[23 * 60 * (num_anchors*(num_classes+5))]
    ---------- process4 ----------
    输入1：
        Conv: [3*3*256] in=y52 strides=2 padding=1 norm=BN active=Mish out=[12 * 30 * 256]
    输入2：
        branch2: out=[12 * 30 * 512]
    Concat: [输入1，输入2] out=[12 * 30 * 768]
    ConvBlock: [3 * 3 * [256,512,256,512,256]] out=[12 * 30 * 256]
    ---------- YOLO HEAD2 ----------
    Conv: [3*3*512] in=process4 strides=1 padding=1 norm=BN active=Mish out=[12 * 30 * 512]
    Conv: [1*1*(num_anchors * (num_classes + 1))] strides=1 padding=0 out=[12 * 30 * (num_anchors*(num_classes+5))]
    ---------- process5 ----------
    输入1：
        Conv: [3*3*512] in=y26 strides=2 padding=1 norm=BN active=Mish out=[6 * 15 * 512]
    输入2：
        y13: out=[6 * 15 * 512]
    Concat: [输入1，输入2] out=[6 * 15 * 1024]
    ConvBlock: [3 * 3 * [512,1024,512,1024,512]] out=[6 * 15 * 512]
    ---------- YOLO HEAD3 ----------
    Conv: [3*3*1024] in=process5 strides=1 padding=1 norm=BN active=Mish out=[6 * 15 * 1024]
    Conv: [1*1*(num_anchors * (num_classes + 1))] strides=1 padding=0 out=[6 * 15 * (num_anchors*(num_classes+5))]
    
    Yolo Hard1: [23 * 60 * (num_anchors*num_classes+5)]
    Yolo Hard2: [12 * 30 * (num_anchors*num_classes+5)]
    Yolo Hard3: [6 * 15 * (num_anchors*num_classes+5)]
@author: luoyi
Created on 2021年2月25日
'''
import tensorflow as tf

import utils.conf as conf
import utils.alphabet as alphabet
from models.layer.commons.part import CSPResBlock, Conv2DNormActive
from models.layer.commons.part import YoloHard, YoloHardRegister, YoloHardRegisterLayer
from models.layer.v4.part import ProcessSPP, ProcessConcatUpSample, ProcessConcatDownSample


#    yolo_v4网络结构
class YoloV4Layer(tf.keras.layers.Layer):
    def __init__(self,
                 name='yolo_v4',
                 input_shape=(conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT, 3),
                 num_anchors=conf.DATASET_CELLS.get_anchors_set().shape[1],
                 num_classes=len(alphabet.ALPHABET),
                 yolohard_register=YoloHardRegister.instance(),
                 **kwargs):
        super(YoloV4Layer, self).__init__(name=name, **kwargs)
        
        #    +2是为了兼容手动padding（模型第一层就上ZeroPadding会存在load_weights的bug）
        self._input_shape = (input_shape[0] + 2, input_shape[1] + 2, input_shape[2])
        
        self._num_anchors=num_anchors
        self._num_classes=num_classes
        
        self._yolohard_register = yolohard_register
        
        self._assembling()
        pass
    
    #    装载网络
    def _assembling(self):
        #    ---------- layer 1 ----------
        #    Conv: [3 * 3 * 32] strides=1 padding=1 norm=BN active=Mish out=[180 * 480 * 32]
        self._layer1 = Conv2DNormActive(name='layer1',
                                        kernel_size=[3, 3], filters=32, strides=1, padding='VALID', 
                                        input_shape=(182, 482, 3), output_shape=(180, 480, 32))
        #    ---------- layer 2 ----------
        #    CSPResBlock: filters=64 times=1 part=[1,1] out=[90 * 240 * 64]
        self._layer2 = CSPResBlock(name='layer2',
                                   filters=64, part_rate=[1, 1], res_block_num=1,
                                   input_shape=(180, 480, 32), output_shape=(90, 240, 64))
        #    ---------- layer 3 ----------
        #    CSPResBlock: filters=128 times=2 part=[0.5, 0.5] out=[45 * 120 * 128]
        self._layer3 = CSPResBlock(name='layer3',
                                   filters=128, part_rate=[0.5, 0.5], res_block_num=2,
                                   input_shape=(90, 240, 64), output_shape=(45, 120, 128))
        #    ---------- layer 4 ----------
        #    CSPResBlock: filters=256 times=8 part=[0.5, 0.5] out=[23 * 60 * 256]
        #    记录：branch1: out=[23 * 60 * 256]
        self._layer4 = CSPResBlock(name='layer4',
                                   filters=256, part_rate=[0.5, 0.5], res_block_num=8,
                                   input_shape=(45, 120, 128), output_shape=(23, 60, 256))
        #    ---------- layer 5 ----------
        #    CSPResBlock: filters=512 times=8 part=[0.5, 0.5] out=[12 * 30 * 512]
        #    记录：branch2: out=[12 * 30 * 512]
        self._layer5 = CSPResBlock(name='layer5',
                                   filters=512, part_rate=[0.5, 0.5], res_block_num=8,
                                   input_shape=(23, 60, 256), output_shape=(12, 30, 512))
        #    ---------- layer 6 ----------
        #    CSPResBlock: filters=1024 times=4 part=[0.5, 0.5] out=[6 * 15 * 1024]
        self._layer6 = CSPResBlock(name='layer6',
                                   filters=1024, part_rate=[0.5, 0.5], res_block_num=4,
                                   input_shape=(12, 30, 512), output_shape=(6, 15, 1024))
        #    ---------- process1 ----------
        #    ConvBlock: [3 * 3 * [512,1024,512]] out=[6 * 15 * 512]
        #    SPP: [6*15, 4*10, 2*5, 1*1] out=[6 * 15 * 4096]
        #    ConvBlock: [3 * 3 * [512,1024,512]] out=[6 * 15 * 512]
        #    记录：y13: out=[6 * 15 * 512]
        self._process1 = ProcessSPP(name='process1',
                                    filters_series1=[512, 1024, 512],
                                    spp_psize=[[6,15], [4,10], [2,5]],
                                    filters_series2=[512, 1024, 512],
                                    input_shape=(6, 15, 1024), output_shape=(6, 15, 512))
        #    ---------- process2 ----------
        #    输入1：
        #        Conv: [1*1*256] in=y13 strides=1 padding=0 norm=BN active=Mish out=[6 * 15 * 256]
        #        up_sample: out=[12 * 30 * 256]
        #    输入2：
        #        branch2: out=[12 * 30 * 512]
        #    Concat: [输入1，输入2] out=[12 * 30 * 768]
        #    ConvBlock: [3 * 3 * [256,512,256,512,256]] out=[12 * 30 * 256]
        #    记录：y26: out=[12 * 30 * 256]
        self._process2 = ProcessConcatUpSample(name='process2',
                                               filters_conv1=256,
                                               filters_series=[256,512,256,512,256],
                                               input_shape1=(6, 15, 512), input_shape2=(12, 30, 512), output_shape=(12, 30, 256))
        #    ---------- process3 ----------
        #    输入1：
        #        Conv: [1*1*128] in=y26 strides=1 padding=0 norm=BN active=Mish out=[12 * 30 * 128]
        #        up_sample: out=[23 * 60 * 128]
        #    输入2：
        #        branch1: out=[23 * 60 * 256]
        #    Concat: [输入1，输入2] out=[23 * 60 * 384]
        #    ConvBlock: [3 * 3 * [128,256,128,256,128]] out=[23 * 60 * 128]
        #    记录：y52: [23 * 60 * 128]
        self._process3 = ProcessConcatUpSample(name='process3',
                                               filters_conv1=128,
                                               filters_series=[128,256,128,256,128],
                                               input_shape1=(12, 30, 256), input_shape2=(23, 60, 256), output_shape=(23, 60, 128))
        #    ---------- process4 ----------
        #    输入1：
        #        Conv: [3*3*256] in=y52 strides=2 padding=1 norm=BN active=Mish out=[12 * 30 * 256]
        #    输入2：
        #        branch2: out=[12 * 30 * 512]
        #    Concat: [输入1，输入2] out=[12 * 30 * 768]
        #    ConvBlock: [3 * 3 * [256,512,256,512,256]] out=[12 * 30 * 256]
        self._process4 = ProcessConcatDownSample(name='process4',
                                                 filters_conv1=256,
                                                 filters_series=[256,512,256,512,256],
                                                 input_shape1=(23, 60, 128), input_shape2=(12, 30, 512), output_shape=(12, 30, 256))
        #    ---------- process5 ----------
        #    输入1：
        #        Conv: [3*3*512] in=y26 strides=2 padding=1 norm=BN active=Mish out=[6 * 15 * 512]
        #    输入2：
        #        y13: out=[6 * 15 * 512]
        #    Concat: [输入1，输入2] out=[6 * 15 * 1024]
        #    ConvBlock: [3 * 3 * [512,1024,512,1024,512]] out=[6 * 15 * 512]
        self._process5 = ProcessConcatDownSample(name='process5',
                                                 filters_conv1=512,
                                                 filters_series=[512,1024,512,1024,512],
                                                 input_shape1=(12, 30, 256), input_shape2=(6, 15, 512), output_shape=(6, 15, 512))
        #    ---------- YOLO HEAD1 ----------
        #    Conv: [3*3*256] in=process3 strides=1 padding=1 norm=BN active=Mish out=[23 * 60 * 256]
        #    Conv: [1*1*(num_anchors * (num_classes + 1))] strides=1 padding=0 out=[23 * 60 * (num_anchors*(num_classes+5))]
        self._yolo_hard1 = YoloHard(name='Yolo_hard1',
                                    filters=256,
                                    num_anchors=self._num_anchors,
                                    num_classes=self._num_classes,
                                    input_shape=(23, 60, 128), output_shape=(23, 60, self._num_anchors, self._num_classes + 5))
        #    ---------- YOLO HEAD2 ----------
        #    Conv: [3*3*512] in=process4 strides=1 padding=1 norm=BN active=Mish out=[12 * 30 * 512]
        #    Conv: [1*1*(num_anchors * (num_classes + 1))] strides=1 padding=0 out=[12 * 30 * (num_anchors*(num_classes+5))]
        self._yolo_hard2 = YoloHard(name='Yolo_hard2',
                                    filters=512,
                                    num_anchors=self._num_anchors,
                                    num_classes=self._num_classes,
                                    input_shape=(12, 30, 256), output_shape=(12, 30, self._num_anchors, self._num_classes + 5))
        #    ---------- YOLO HEAD3 ----------
        #    Conv: [3*3*1024] in=process5 strides=1 padding=1 norm=BN active=Mish out=[6 * 15 * 1024]
        #    Conv: [1*1*(num_anchors * (num_classes + 1))] strides=1 padding=0 out=[6 * 15 * (num_anchors*(num_classes+5))]
        self._yolo_hard3 = YoloHard(name='Yolo_hard3',
                                    filters=1024,
                                    num_anchors=self._num_anchors,
                                    num_classes=self._num_classes,
                                    input_shape=(6, 15, 512), output_shape=(6, 15, self._num_anchors, self._num_classes + 5))
        
        #    Yolo hard寄存器
        self._yolo_hard_register_layer = YoloHardRegisterLayer(yolohard_register=self._yolohard_register)
        pass
    
    #    前向传播
    def call(self, x, training=None, **kwargs):
        #    先做padding1（放模型第一层会存在load_weight的bug，所以只能放这）
        x = tf.pad(x, paddings=[[0,0], [1,1], [1,1], [0,0]])
        
        #    CSPDarknet53逻辑
        y = self._layer1(x)
        y = self._layer2(y)
        y = self._layer3(y)
        branch1 = y = self._layer4(y)       #    [23 * 60 * 256]
        branch2 = y = self._layer5(y)       #    [12 * 30 * 512]
        y = self._layer6(y)                 #    [6 * 15 * 1024]
        
        #    SPP + PAN
        process1 = self._process1(y)                                #    [6 * 15 * 512]
        process2 = self._process2(x1=process1, x2=branch2)          #    [12 * 30 * 256]
        process3 = self._process3(x1=process2, x2=branch1)          #    [23 * 60 * 128]
        process4 = self._process4(x1=process3, x2=branch2)          #    [12 * 30 * 256]
        process5 = self._process5(x1=process2, x2=process1)         #    [6 * 15 * 512]
        
        #    yolo hard
        yolo_hard1 = self._yolo_hard1(process3)         #    [23 * 60 * (num_anchors*(num_classes+5))]
        yolo_hard2 = self._yolo_hard2(process4)         #    [12 * 30 * (num_anchors*(num_classes+5))]
        yolo_hard3 = self._yolo_hard3(process5)         #    [6 * 15 * (num_anchors*(num_classes+5))]
        
        #    寄存3个yolohard输出
        self._yolo_hard_register_layer(x=x, yolohard1=yolo_hard1, yolohard2=yolo_hard2, yolohard3=yolo_hard3)
        return yolo_hard3
    pass


