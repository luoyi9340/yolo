# -*- coding: utf-8 -*-  
'''
yolo v4版本零件

    - process1
        ConvBlock: [3 * 3 * [512,1024,512]] out=[6 * 15 * 512]
        SPP: [6*15, 4*10, 2*5, 1*1] out=[6 * 15 * 4096]
        ConvBlock: [3 * 3 * [512,1024,512]] out=[6 * 15 * 512]
        记录：y13: out=[6 * 15 * 512]
        
    - process2
        输入1：
            Conv: [1*1*256] in=y13 strides=1 padding=0 norm=BN active=Mish out=[6 * 15 * 256]
            up_sample: out=[12 * 30 * 256]
        输入2：
            branch2: out=[12 * 30 * 512]
        Concat: [输入1，输入2] out=[12 * 30 * 768]
        ConvBlock: [3 * 3 * [256,512,256,512,256]] out=[12 * 30 * 256]
        记录：y26: out=[12 * 30 * 256]
        
    - process3
        输入1：
            Conv: [1*1*128] in=y26 strides=1 padding=0 norm=BN active=Mish out=[12 * 30 * 128]
            up_sample: out=[23 * 60 * 128]
        输入2：
            branch1: out=[23 * 60 * 256]
        Concat: [输入1，输入2] out=[23 * 60 * 384]
        ConvBlock: [3 * 3 * [128,256,128,256,128]] out=[23 * 60 * 128]
        
    - process4
        输入1：
            Conv: [3*3*256] in=y52 strides=2 padding=1 norm=BN active=Mish out=[12 * 30 * 256]
        输入2：
            branch2: out=[12 * 30 * 512]
        Concat: [输入1，输入2] out=[12 * 30 * 768]
        ConvBlock: [3 * 3 * [256,512,256,512,256]] out=[12 * 30 * 256]
        
    - process5
        输入1：
            Conv: [3*3*512] in=y26 strides=2 padding=1 norm=BN active=Mish out=[6 * 15 * 512]
        输入2：
            y13: out=[6 * 15 * 512]
        Concat: [输入1，输入2] out=[6 * 15 * 1024]
        ConvBlock: [3 * 3 * [512,1024,512,1024,512]] out=[6 * 15 * 512]
    
    
    - yolo_hard1
        Conv: [3*3*256] in=process3 strides=1 padding=1 norm=BN active=Mish out=[23 * 60 * 256]
        Conv: [1*1*(num_anchors * (num_classes + 1))] strides=1 padding=0 out=[23 * 60 * (num_anchors*(num_classes+5))]
        
    - yolo_hard2
        Conv: [3*3*512] in=process4 strides=1 padding=1 norm=BN active=Mish out=[12 * 30 * 512]
        Conv: [1*1*(num_anchors * (num_classes + 1))] strides=1 padding=0 out=[12 * 30 * (num_anchors*(num_classes+5))]
        
    - yolo_hard3
        Conv: [3*3*1024] in=process5 strides=1 padding=1 norm=BN active=Mish out=[6 * 15 * 1024]
        Conv: [1*1*(num_anchors * (num_classes + 1))] strides=1 padding=0 out=[6 * 15 * (num_anchors*(num_classes+5))]

@author: luoyi
Created on 2021年2月25日
'''
import tensorflow as tf

from models.layer.commons.part import Conv2DNormActive
from models.layer.commons.part import SeriesConv2D, SPPBlock
from models.layer.commons.part import UpSampling, UpSamplingOpType
from math import ceil


#    ProcessSPP
class ProcessSPP(tf.keras.layers.Layer):
    '''
        ConvBlock: [3 * 3 * [512,1024,512]] out=[6 * 15 * 512]
        SPP: [6*15, 4*10, 2*5, 1*1] out=[6 * 15 * 4096]
        ConvBlock: [3 * 3 * [512,1024,512]] out=[6 * 15 * 512]
        记录：y13: out=[6 * 15 * 512]
    '''
    def __init__(self,
                 name='ProcessSPP',
                 filters_series1=[512,1024,512],
                 filters_series2=[512,1024,512],
                 spp_psize=[[6,15], [4,10], [2,5]],
                 input_shape=None,
                 output_shape=None,
                 **kwargs):
        super(ProcessSPP, self).__init__(name=name, **kwargs)
        
        self._input_shape = input_shape
        self._output_shape = output_shape
        
        #    装配网络
        self._layer = tf.keras.models.Sequential([
                        SeriesConv2D(name=name + '_ConvBlock1', filters=filters_series1),
                        SPPBlock(name=name + '_SPPBlock', pool_sizes=spp_psize),
                        SeriesConv2D(name=name + '_ConvBlock2', filters=filters_series2)
                    ], name=name + '_layer')
        pass
    
    def call(self, x, **kwargs):
        #    验证输入尺寸
        if (self._input_shape
                and self._input_shape != x.shape[1:]):
            raise Exception(self.name + " input_shape:" + str(self._input_shape) + " not equal x:" + str(x.shape))
        
        y = self._layer(x)
        
        #    验证输出尺寸
        if (self._output_shape
                and self._output_shape != y.shape[1:]):
            raise Exception(self.name + " output_shape:" + str(self._output_shape) + " not equal y:" + str(y.shape))
        
        return y
    pass


#    ProcessConcatUpSample
class ProcessConcatUpSample(tf.keras.layers.Layer):
    ''' 合并输出，输入1上采样
        输入1：
            Conv: [1*1*filters_conv1 in=y13 strides=1 padding=0 norm=BN active=Mish out=[H1 * W1 * filters_conv1]
            up_sample: out=[2*H1 * 2*H2 * filters_conv1]
        输入2：
            branch2: out=[H2=2*H1 * W2=2*H1 * C2]
        Concat: [输入1，输入2] out=[H2 * W2 * (filters_conv1+C2)]
        ConvBlock: [3 * 3 * [f1,f2,f3,f4,f5]] out=[H2 * W2 * f5]
        记录：y26: out=[H2 * W2 * f5]
    '''
    def __init__(self,
                 name='ProcessConcatUpSample',
                 filters_conv1=16,
                 filters_series=[512,1024,512,1024,512],
                 input_shape1=None,
                 input_shape2=None,
                 output_shape=None,
                 **kwargs):
        #    验证input_shape2的尺寸必须是input_shape1的2倍
        assert (ceil(input_shape2[0] / input_shape1[0]) == 2 
                and ceil(input_shape2[1] / input_shape1[1]) == 2), \
                'input_shape2.shape must twice as much as input_shape1. input_shape2' + str(input_shape2) + ' input_shape1:' + str(input_shape1)
        
        super(ProcessConcatUpSample, self).__init__(name=name, **kwargs)
        
        self._input_shape1 = input_shape1
        self._input_shape2 = input_shape2
        self._output_shape = output_shape
        
        #    输入1
#         self._branch1 = tf.keras.models.Sequential(name=name + '_branch1')
#         self._branch1.add(Conv2DNormActive(name=name + '_conv11_branch1',
#                                            kernel_size=[1,1],
#                                            filters=filters_conv1,
#                                            strides=1,
#                                            padding='VALID',
#                                            input_shape=input_shape1))
#         self._branch1.add(UpSampling(name=name + '_upsmapling',
#                                      op_type=UpSamplingOpType.BiLinearInterpolation,
#                                      input_shape=(input_shape1[0], input_shape1[1], filters_conv1),
#                                      output_shape=(output_shape[0], output_shape[1], filters_conv1)))
        self._branch1_conv = Conv2DNormActive(name=name + '_conv11_branch1',
                                              kernel_size=[1,1],
                                              filters=filters_conv1,
                                              strides=1,
                                              padding='VALID',
                                              input_shape=input_shape1)
        self._branch1_upsample = UpSampling(name=name + '_upsmapling',
                                            op_type=UpSamplingOpType.BiLinearInterpolation,
                                            input_shape=(input_shape1[0], input_shape1[1], filters_conv1),
                                            output_shape=(output_shape[0], output_shape[1], filters_conv1))
        
        #    一连串Conv
        self._series_conv = SeriesConv2D(name=name + '_SeriesConv2D',
                                         filters=filters_series)
        pass
    
    def call(self, x1=None, x2=None, **kwargs):
        #    验证输入尺寸
        if (self._input_shape1
                and self._input_shape1 != x1.shape[1:]):
            raise Exception(self.name + " input_shape1:" + str(self._input_shape1) + " not equal x:" + str(x1.shape))  
        if (self._input_shape2
                and self._input_shape2 != x2.shape[1:]):
            raise Exception(self.name + " input_shape2:" + str(self._input_shape2) + " not equal x:" + str(x2.shape))           
        
        #    x1过分支1
#         x1 = self._branch1(x1)
        x1 = self._branch1_conv(x1)
        x1 = self._branch1_upsample(x1)

        #    结果与x2叠加
        y = tf.concat([x1, x2], axis=-1)
        
        #    一连串卷积
        y = self._series_conv(y)
        
        #    验证输出尺寸
        if (self._output_shape
                and self._output_shape != y.shape[1:]):
            raise Exception(self.name + " output_shape:" + str(self._output_shape) + " not equal y:" + str(y.shape))
        
        return y
    pass


#    ProcessConcatDownSample
class ProcessConcatDownSample(tf.keras.layers.Layer):
    '''合并输出，并且输入1下采样
        输入1：
            Conv: [3*3*filters_conv1] in=[2H * 2W * C1] strides=2 padding=1 norm=BN active=Mish out=[H * W * filters_conv1]
        输入2：
            y13: out=[H * W * C2]
        Concat: [输入1，输入2] out=[H * W * (filters_conv1 + C2)]
        ConvBlock: [3 * 3 * [f1, f2, f3, f4, f5]] out=[H * W * f5]
    '''
    def __init__(self,
                 name='ProcessConcatDownSample',
                 filters_conv1=16,
                 filters_series=[512,1024,512,1024,512],
                 input_shape1=None,
                 input_shape2=None,
                 output_shape=None,
                 **kwargs):
        #    验证input_shape1的尺寸必须是input_shape2的2倍
        assert (ceil(input_shape1[0] / input_shape2[0]) == 2 
                 and ceil(input_shape1[1] / input_shape2[1]) == 2), \
                'input_shape1.shape must twice as much as input_shape2. input_shape1' + str(input_shape1) + ' input_shape2:' + str(input_shape2)
        
        super(ProcessConcatDownSample, self).__init__(name=name, **kwargs)
        
        self._input_shape1 = input_shape1
        self._input_shape2 = input_shape2
        self._output_shape = output_shape
        
        #    输入1
        self._branch1 = tf.keras.models.Sequential(name=name + '_branch1')
        self._branch1.add(Conv2DNormActive(name=name + '_conv11_branch1',
                                           kernel_size=[3,3],
                                           filters=filters_conv1,
                                           strides=2,
                                           padding=1,
                                           input_shape=input_shape1))
        
        #    一连串Conv
        self._series_conv = SeriesConv2D(name=name + '_SeriesConv2D',
                                         filters=filters_series)
        pass
    
    def call(self, x1=None, x2=None, **kwargs):
        #    验证输入尺寸
        if (self._input_shape1
                and self._input_shape1 != x1.shape[1:]):
            raise Exception(self.name + " input_shape1:" + str(self._input_shape1) + " not equal x:" + str(x1.shape))  
        if (self._input_shape2
                and self._input_shape2 != x2.shape[1:]):
            raise Exception(self.name + " input_shape2:" + str(self._input_shape2) + " not equal x:" + str(x2.shape))         
        
        #    x1过分支1
        x1 = self._branch1(x1)
        
        #    结果与x2叠加
        y = tf.concat([x1, x2], axis=-1)
        
        #    一连串卷积
        y = self._series_conv(y)
        
        #    验证输出尺寸
        if (self._output_shape
                and self._output_shape != y.shape[1:]):
            raise Exception(self.name + " output_shape:" + str(self._output_shape) + " not equal y:" + str(y.shape))
        
        return y
    pass

