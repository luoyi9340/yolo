# -*- coding: utf-8 -*-  
'''
v4-tiny 需要的组件

    - CSPBlock
        与V4中的CSPResBLock块类似，区别在这里去掉了残差相加，只保留残差堆叠。也没有重复次数。下采样改为maxpooling实现。分支也改为直接通道维度裁剪
        输入参数：filters=输出通道数
        具体结构为：
            输入: [H * W * C1]
            ------ 3*3卷积核特征整合 ------
            Conv: [3 * 3 * filters/2] strides=1 padding=1 norm=BN active=LeakyReLU out=[H * W * filters/2]
            ------ 分支 ------
            part1 = 上层输出 out=[H * W * filters/2]
            part2 = 裁剪前通道后半部分 out=[H * W * filters/4]
                part2_p1 = Conv: [3 * 3 * filters/4] in=part2 strides=1 padding=1 norm=BN active=LeakyReLU out=[H * W * filters/4]
                part2_p2 = Conv: [3 * 3 * filters/4] in=par2_p1 strides=1 padding=1 norm=BN active=LeakyReLU out=[H * W * filters/4]
                通道维度堆叠part2_p1与part2_p2: out=[H * W * filters/2]
                part2 = Conv: [3 * 3 * filters/2] in=par2_p1 strides=1 padding=1 norm=BN active=LeakyReLU out=[H * W * filters/2]
            通道维度堆叠part1与part2L out=[H * W * filters]
            ------ maxpooling ------
            maxpooling: [2*2] stride=1 padding=0 out=[H/2 * W/2 * filters]
    
    - CSPUpSampleBlock
        通过上采样整合两部分输入的形状，拼接在一起。不改变通道数
        输入参数：filters=输出通道数
        具体结构为：
            输入：两部分输入
            ------ 分支 ------
            分支1：输入=branch1 = [2H * 2W * filters]
            分支2：输入=branch2 = [H * W * filters]
                Conv: [1 * 1 * C/2] strides=1 padding=0 norm=BN active=LeakyReLU out=[H * W * filters/2]
                UpSample: out=[2H * 2W * filters/2]
            Concat: out=[2H * 2W * (filters + filters/2)]
            Conv: [3 * 3 * filters] strides=1 padding=1 norm=BN active=LeakyReLU out=[2H * 2W * filters]
    

@author: luoyi
Created on 2021年3月9日
'''
import tensorflow as tf

import utils.alphabet as alphabet
from models.layer.commons.part import Conv2DNormActive
from models.layer.commons.part import UpSampling, UpSamplingOpType
from math import ceil


#    CSPBlock
class CSPBlockTinyLayer(tf.keras.layers.Layer):
    '''
            输入: [H * W * C1]
            ------ 3*3卷积核特征整合 ------
            Conv: [3 * 3 * filters/2] strides=1 padding=1 norm=BN active=LeakyReLU out=[H * W * filters/2]
            ------ 分支 ------
            part1 = 上层输出 out=[H * W * filters/2]
            part2 = 裁剪前通道后半部分 out=[H * W * filters/4]
                part2_p1 = Conv: [3 * 3 * filters/4] in=part2 strides=1 padding=1 norm=BN active=LeakyReLU out=[H * W * filters/4]
                part2_p2 = Conv: [3 * 3 * filters/4] in=par2_p1 strides=1 padding=1 norm=BN active=LeakyReLU out=[H * W * filters/4]
                通道维度堆叠part2_p1与part2_p2: out=[H * W * filters/2]
                part2 = Conv: [1 * 1 * filters/2] in=par2_p1 strides=1 padding=1 norm=BN active=LeakyReLU out=[H * W * filters/2]
            通道维度堆叠part1与part2L out=[H * W * filters]
            ------ maxpooling ------
            maxpooling: [2*2] stride=1 padding=0 out=[H/2 * W/2 * filters]
    '''
    def __init__(self,
                 name='CSPBlockTinyLayer',
                 filters=16,
                 input_shape=None,
                 output_shape=None,
                 pooling_padding=None,
                 **kwargs):
        '''
            @param filters: 输出通道数
            @param pooling_padding: maxpooling层是否需要padding
        '''
        super(CSPBlockTinyLayer, self).__init__(name=name, **kwargs)
        
        half_filters = int(filters / 2)
        double_half_filters = int(filters / 4)
        self._half_filters = half_filters
        self._double_half_filters = double_half_filters
        
        self._input_shape = input_shape
        self._output_shape = output_shape
        
        self._pooling_padding = pooling_padding
        
        #    第一部分卷积
        self._conv1 = Conv2DNormActive(name=name + '_conv1',
                                       filters=half_filters,
                                       kernel_size=[3, 3],
                                       strides=1,
                                       padding='SAME',
                                       active=tf.keras.layers.LeakyReLU(),
                                       )
        
        #    第二部分两个卷积
        self._conv21 = Conv2DNormActive(name=name + '_conv21',
                                        filters=double_half_filters,
                                        kernel_size=[3, 3],
                                        strides=1,
                                        padding='SAME',
                                        active=tf.keras.layers.LeakyReLU(),
                                        )
        self._conv22 = Conv2DNormActive(name=name + '_conv22',
                                        filters=double_half_filters,
                                        kernel_size=[3, 3],
                                        strides=1,
                                        padding='SAME',
                                        active=tf.keras.layers.LeakyReLU(),
                                        )
        #    第三部分卷积
        self._conv3 = Conv2DNormActive(name=name + '_conv3',
                                       filters=half_filters,
                                       kernel_size=[1, 1],
                                       strides=1,
                                       padding='SAME',
                                       active=tf.keras.layers.LeakyReLU(),
                                       )
        
        #    maxpooling层
        self._maxpooling = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='VALID')
        pass
    
    def call(self, x, **kwargs):
        #    验证输入格式
        if (self._input_shape
                and self._input_shape != x.shape[1:]):
            raise Exception(self.name + " input_shape:" + str(self._input_shape) + " not equal x:" + str(x.shape))
        
        #    第一部分卷积
        y = self._conv1(x)
        
        #    分支
        #    part1取第一部分卷积结果
        part1 = y
        #    part2取通道维度后半部分
        [_, part2] = tf.split(y, num_or_size_splits=[self._double_half_filters, self._double_half_filters], axis=-1)
        
        #    part2过卷积
        part2_p1 = self._conv21(part2)
        part2_p2 = self._conv22(part2_p1)
        #    堆叠part2
        part2 = tf.concat([part2_p1, part2_p2], axis=-1)
        
        #    part2过第三部分卷积
        part2 = self._conv3(part2)
        
        #    堆叠part1与part2
        y = tf.concat([part1, part2], axis=-1)
        
        #    过max_pooling层
        if (self._pooling_padding is not None):
            y = tf.pad(y, paddings=[[0,0], 
                                    [self._pooling_padding[0], self._pooling_padding[0]], 
                                    [self._pooling_padding[1], self._pooling_padding[1]], 
                                    [0,0]])
            pass
        y = self._maxpooling(y)
        
        #    验证输出结果
        if (self._output_shape
                and self._output_shape != y.shape[1:]):
            raise Exception(self.name + " output_shape:" + str(self._output_shape) + " not equal y:" + str(y.shape))
        
        return y
    
    pass


#    CSPUpSampleBlock
class CSPUpSampleBlockTinyLayer(tf.keras.layers.Layer):
    '''
        通过上采样整合两部分输入的形状，拼接在一起。不改变通道数
        输入参数：filters=输出通道数
        具体结构为：
            输入：两部分输入
            ------ 分支 ------
            分支1：输入=branch1 = [2H * 2W * filters]
            分支2：输入=branch2 = [H * W * filters]
                Conv: [1 * 1 * C/2] strides=1 padding=0 norm=BN active=LeakyReLU out=[H * W * filters/2]
                UpSample: out=[2H * 2W * filters/2]
            Concat: out=[2H * 2W * (filters + filters/2)]
            Conv: [3 * 3 * filters] strides=1 padding=1 norm=BN active=LeakyReLU out=[2H * 2W * filters]
    '''
    def __init__(self,
                 name='CSPUpSampleBlockTinyLayer',
                 filters=16,
                 input_shape1=None,
                 input_shape2=None,
                 output_shape=None,
                 **kwargs):
        #    验证input_shape1的尺寸必须是input_shape2的2倍
        assert (ceil(input_shape1[0] / input_shape2[0]) == 2 
                 and ceil(input_shape1[1] / input_shape2[1]) == 2), \
                'input_shape1.shape must twice as much as input_shape2. input_shape1' + str(input_shape1) + ' input_shape2:' + str(input_shape2)
        
        super(CSPUpSampleBlockTinyLayer, self).__init__(name=name, **kwargs)
        
        half_filters = int(filters / 2)
        
        self._input_shape1 = input_shape1
        self._input_shape2 = input_shape2
        self._output_shape = output_shape
        
        #    branch2卷积层
        self._branch2_conv = Conv2DNormActive(name=name + '_branch2_conv',
                                              filters=half_filters,
                                              kernel_size=[1, 1],
                                              strides=1,
                                              padding='SAME',
                                              active=tf.keras.layers.LeakyReLU())
        #    branch2上采样层
        self._branch2_upsample = UpSampling(name=name + '_branch2_upsample',
                                            op_type=UpSamplingOpType.BiLinearInterpolation,
                                            input_shape=(input_shape2[0], input_shape2[1], half_filters),
                                            output_shape=(2 * input_shape2[0], 2 * input_shape2[1], half_filters))
        
        #    合并后的卷积层
        self._conv = Conv2DNormActive(name=name + '_conv',
                                      filters=filters,
                                      kernel_size=[3, 3],
                                      strides=1,
                                      padding='SAME',
                                      active=tf.keras.layers.LeakyReLU())
        pass
    
    def call(self, x, x1=None, x2=None, **kwargs):
        #    验证输入尺寸
        if (self._input_shape1
                and self._input_shape1 != x1.shape[1:]):
            raise Exception(self.name + " input_shape1:" + str(self._input_shape1) + " not equal x:" + str(x1.shape))  
        if (self._input_shape2
                and self._input_shape2 != x2.shape[1:]):
            raise Exception(self.name + " input_shape2:" + str(self._input_shape2) + " not equal x:" + str(x2.shape))  
        
        #    输入2过卷积层
        x2 = self._branch2_conv(x2)
        #    输入2过上采样层
        x2 = self._branch2_upsample(x2)
        
        #    合并x1与x2
        y = tf.concat([x1, x2], axis=-1)
        
        #    过合并后的卷积层
        y = self._conv(y)
        
        #    验证输出尺寸
        if (self._output_shape
                and self._output_shape != y.shape[1:]):
            raise Exception(self.name + " output_shape:" + str(self._output_shape) + " not equal y:" + str(y.shape))   
        
        return y
    pass


#    Yolohard过激活函数
class YolohardActiveLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 name='YolohardActiveLayer',
                 num_classes=len(alphabet.ALPHABET),
                 **kwargs):
        super(YolohardActiveLayer, self).__init__(name=name, trainable=False, **kwargs)
        
        self._num_classes = num_classes
        pass
    def call(self, x, **kwargs):
        '''
            @param x: Tensor(batch_size, H, W, num_anchors, num_classes + 5)
        '''
        #    分类预测    Tensor(batch_size, H, W, num_anchors, num_classes)
        yolohard_cls = x[:, :,:, :, :self._num_classes]
        yolohard_cls = tf.nn.softmax(yolohard_cls)
        #    预测置信度    Tensor(batch_size, H, W, num_anchors, 1)
        yolohard_confidence = x[:, :,:, :, self._num_classes]
        yolohard_confidence = tf.expand_dims(yolohard_confidence, axis=-1)
        yolohard_confidence = tf.nn.sigmoid(yolohard_confidence)
        #    预测dx,dy    Tensor(batch_size, H, W, num_anchors, 2)
        yolohard_dxy = x[:, :,:, :, self._num_classes + 1 : self._num_classes + 3]
        yolohard_dxy = tf.nn.sigmoid(yolohard_dxy)
        #    预测dw,dh    Tensor(batch_size, H, W, num_anchors, 2)
        yolohard_dwh = x[:, :,:, :, self._num_classes + 3:]
        
        #    组合
        yolohard = tf.concat([yolohard_cls, 
                              yolohard_confidence,
                              yolohard_dxy,
                              yolohard_dwh], axis=-1)
        return yolohard
    pass


