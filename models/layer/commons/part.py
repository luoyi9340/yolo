# -*- coding: utf-8 -*-  
'''
yolo 需要用到的一些公共零件

    - Mish激活函数
    
    - Conv2DNormActive
        Conv + norm + BN + active
        
    - SeriesConv2D
        Conv[1*1] + Conv[3*3] + Conv[1*1] + ...
    
    - CSPResBlock
        part1: Conv[1*1*C/2]
        part2: Conv[1*1*C/2]
                ResBlock
        Concat:
        Conv[1*1]
        
    - ResBlock
        [Conv[1*1] + Conv[3*3] + 残差快] * num
        
    - SPPBlock
        p1 = maxpool: [H1*W1]
        p2 = maxpool: [H2*W2]
        p3 = maxpool: [H3*W3]
        ...
        Concat: p1 + p2 + p3 + ... + X, axis=-1
        
    - UpSampling
        双线性插值
        Reshape
        转置卷积
        直接填充
        
    - YoloHard
        Conv[3*3]
        Conv[1*1 * num_anchors * (num_classes + 5)]
        Reshape: [batch_size, H, W, num_anchors, sigmoid(num_classes) + sigmoid(1) + sigmoid(2) + 2]
                    sigmoid(num_classes): 每个cell的每个anchor的 分类预测（不用Softmax是为了兼容一个物体同时属于多分类）
                    sigmoid(1): 每个cell的每个anchor的 置信度预测
                    sigmoid(2): 每个cell的每个anchor的 x,y偏移量预测
                    2: 每个cell的每个anchor的 宽高缩放预测
                    
    - YoloHardRegister（YoloHard寄存器）
        暂存YoloHard 3种尺寸的结果，用于后续的losses和metrics
        YoloHard1
        YoloHard2
        YoloHard3

@author: luoyi
Created on 2021年2月24日
'''
from enum import Enum
import threading

import tensorflow as tf
import utils.conf as conf
from math import ceil, floor


#    Mish激活函数
class Mish(tf.keras.layers.Layer):
    ''' Mish激活函数
        Mish(x) = x * tanh(ln(1 + exp(x)))
    '''    
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        pass
    
    #    计算Mish(x) = x * tanh(ln(1 + exp(x)))
    def call(self, x):
        x = tf.math.log(1 + tf.math.exp(x))
        return x * tf.nn.tanh(x)
    pass


#    ConvNormActive
class Conv2DNormActive(tf.keras.layers.Layer):
    '''
        Conv2D -> BN -> active
    '''
    def __init__(self, 
                 name='Conv2DNormActive', 
                 trainable=True,
                 kernel_size=[3, 3],
                 filters=16,
                 strides=1,
                 padding='VALID',
                 input_shape=None,
                 output_shape=None,
                 norm=None,
                 active=None,
                 kernel_initializer=tf.initializers.he_normal(),
                 bias_initializer=tf.initializers.zeros(),
                 **kwargs):
        super(Conv2DNormActive, self).__init__(name=name, **kwargs)
        
        self._input_shape = input_shape
        self._output_shape = output_shape
        
        self._layer = tf.keras.models.Sequential(name=name + '_layer')
        
        #    Conv层
        if (isinstance(padding, int)):
            self._layer.add(tf.keras.layers.ZeroPadding2D(padding=padding))
            padding = 'VALID'
            pass
        self._layer.add(tf.keras.layers.Conv2D(name=name + '_Conv',
                                               kernel_size=kernel_size,
                                               filters=filters,
                                               strides=strides,
                                               padding=padding,
                                               trainable=trainable,
                                               kernel_initializer=kernel_initializer,
                                               bias_initializer=bias_initializer))
        #    BN层
        if (norm is None): norm = tf.keras.layers.BatchNormalization()
        self._layer.add(norm)
        #    激活层
        if (active is None): active = Mish()
        self._layer.add(active)
        pass
    
    #    前向
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


#    一连串卷积
class SeriesConv2D(tf.keras.layers.Layer):
    ''' 一连串卷积，不改变尺寸
        1*1*filters[0] -> 3*3*filters[1] -> 1*1*filters[2] ...
    '''
    def __init__(self,
                 name='SeriesConv2D',
                 filters=[8, 16, 8],
                 **kwargs):
        super(SeriesConv2D, self).__init__(name=name, **kwargs)
        
        self._layer = tf.keras.models.Sequential(name=name + '_layer')
        
        for i, num_filter in enumerate(filters):
            self._layer.add(Conv2DNormActive(name=name + '_Conv' + str(i),
                                             filters=num_filter,
                                             kernel_size=[1,1] if i % 2 == 0 else [3, 3],
                                             strides=1,
                                             padding='SAME'))
            
            pass
        pass
    
    def call(self, x, **kwargs):
        y = self._layer(x)
        return y
    pass


#    CSPResBlock
class CSPResBlock(tf.keras.layers.Layer):
    def __init__(self,
                 name='CSPResBlock',
                 part_rate=[0.5, 0.5],
                 res_block_num=3,
                 filters=32,
                 input_shape=None,
                 output_shape=None,
                 **kwargs):
        super(CSPResBlock, self).__init__(name=name, **kwargs)
        
        self._input_shape = input_shape
        self._output_shape = output_shape
        
        #    先过3*3卷积核，让尺寸缩放到一半
        self._conv33 = Conv2DNormActive(name=name + '_Conv33',
                                        filters=filters,
                                        kernel_size=[3, 3],
                                        strides=2,
                                        padding=1
                                        )
        #    两个1*1卷积核，用于切分part1,part2
        filter_part1 = ceil(filters * part_rate[0])
        self._conv_cut1 = Conv2DNormActive(name=name + '_Conv_cut1',
                                           filters=filter_part1,
                                           kernel_size=[1, 1],
                                           strides=1,
                                           padding='SAME'
                                           )
        filter_part2 = floor(filters * part_rate[1])
        self._conv_cut2 = Conv2DNormActive(name=name + '_Conv_cut2',
                                           filters=filter_part2,
                                           kernel_size=[1, 1],
                                           strides=1,
                                           padding='SAME'
                                           )
        #    原ResBlock逻辑
        self._res_block = ResBlock(name=name + '_ResBlock',
                                   filters=filter_part2,
                                   num=res_block_num)
        
        #    1*1卷积核规范part2输出通道
        self._conv11_part2 = Conv2DNormActive(name=name + '_Conv11_part2',
                                              filters=filter_part2,
                                              kernel_size=[1, 1],
                                              strides=1,
                                              padding='SAME'
                                              )
        #    1*1卷积核规范输出通道
        self._conv11 = Conv2DNormActive(name=name + '_Conv11',
                                        filters=filters,
                                        kernel_size=[1, 1],
                                        strides=1,
                                        padding='SAME'
                                        )
        pass
    
    def call(self, x, **kwargs):
        #    验证输入尺寸
        if (self._input_shape
                and self._input_shape != x.shape[1:]):
            raise Exception(self.name + " input_shape:" + str(self._input_shape) + " not equal x:" + str(x.shape))
        
        #    过3*3卷积核，特征图尺寸缩小1倍
        y = self._conv33(x)
        
        #    过1*1卷积核，输入分为两支
        part1 = self._conv_cut1(y)
        part2 = self._conv_cut2(y)
        
        #    part2过ResBlock逻辑
        part2 = self._res_block(part2)
        #    part2过1*1卷积核，规范输出通道数
        part2 = self._conv11_part2(part2)
        
        #    合并[part1, part2]
        y = tf.concat([part1, part2], axis=-1)
        
        #    合并结果过1*1卷积核，规范输出通道数
        y = self._conv11(y)
        
        #    验证输出尺寸
        if (self._output_shape
                and self._output_shape != y.shape[1:]):
            raise Exception(self.name + " output_shape:" + str(self._output_shape) + " not equal y:" + str(y.shape))
        
        return y
    pass


#    ResBlock块
class ResBlock(tf.keras.layers.Layer):
    '''残差快
        [1*1卷积核 -> 3*3卷积核] * num， 每个1*1 + 3*3 后追加上一步的残差
        残差快不改变特征图尺寸
    '''
    def __init__(self,
                 name='ResBlock',
                 num=3,
                 filters=16,
                 **kwargs):
        super(ResBlock, self).__init__(name=name, **kwargs)
        
        #    初始化残差快
        self._blocks = []
        for i in range(num):
            block = tf.keras.models.Sequential([Conv2DNormActive(name=name + '_Conv11_' + str(i),
                                                                 filters=filters,
                                                                 kernel_size=[1, 1],
                                                                 strides=1,
                                                                 padding='SAME'),
                                                Conv2DNormActive(name=name + '_Conv33_' + str(i),
                                                                 filters=filters,
                                                                 kernel_size=[3, 3],
                                                                 strides=1,
                                                                 padding='SAME'),
                                            ],name=name + '_ResBlock' + str(i))
            self._blocks.append(block)
            pass
        pass
    
    def call(self, x, **kwargs):
        #    连续执行block，并且每次执行完后追加残差
        for block in self._blocks:
            y = block(x)
            x = tf.keras.layers.add([x, y])
            pass
        return x
    pass


#    SPPBlock
class SPPBlock(tf.keras.layers.Layer):
    def __init__(self,
                 name='SPPBlock',
                 pool_sizes=[[13,13], [9,9], [5,5]],
                 **kwargs):
        super(SPPBlock, self).__init__(name=name, **kwargs)
        
        self._poolings = []
        for ps in pool_sizes:
            pooling = tf.keras.layers.MaxPool2D(pool_size=ps, strides=1, padding='SAME')
            self._poolings.append(pooling)
            pass
        pass
    
    def call(self, x, **kwargs):
        y = [x]
        for pooling in self._poolings:
            y.append(pooling(x))
            pass
        
        y = tf.concat(y, axis=-1)
        return y
    pass


#    上采样类型
class UpSamplingOpType(Enum):
    #    双线性插值
    BiLinearInterpolation = 0
    #    重排（H/2 * W/2 * C*4 -> H*W*C）
    Reshape = 1
    #    转置卷积（转置卷积仅仅是形状映射）
    TransposedConv = 2
    #    填充（下层1个像素映射为上层[strides[0] * strides[1]]的区域）
    Filling = 3
    pass


#    上采样
class UpSampling(tf.keras.layers.Layer):
    def __init__(self,
                 name='UnSampling',
                 strides=[2,2],
                 op_type=UpSamplingOpType.BiLinearInterpolation,
                 input_shape=None,
                 output_shape=None,
                 **kwargs):
        super(UpSampling, self).__init__(name=name, trainable=False, **kwargs)
        
        self._op_type = op_type
        self._strides = strides
        
        self._input_shape = input_shape
        self._output_shape = output_shape
        
        #    如果是reshape，则提前初始化reshape层
        if (op_type == UpSamplingOpType.Reshape):
            H, W, C = input_shape[1], input_shape[2], input_shape[-1]
            S = self._strides
            self._reshape = tf.keras.layers.Reshape(target_shape=(H * S[0], W * S[1], int(C / (S[0] * S[1]))))
            pass
        #    如果是转置卷积，则提前初始化转置卷积核
        if (op_type == UpSamplingOpType.TransposedConv):
            self._transpose_conv = tf.keras.layers.Conv2DTranspose(filters=input_shape[-1],
                                                                   kernel_size=[3, 3], 
                                                                   strides=2, 
                                                                   padding='SAME',
                                                                   kernel_initializer=tf.initializers.he_normal(),
                                                                   bias_initializer=tf.initializers.zeros())
            pass
        pass

    def compute_output_shape(self, input_shape):
        #    高度，宽度，通道数
        H, W, C = input_shape[1], input_shape[2], input_shape[-1]
        S = self._strides
        
        #    双线性插值，改变H,W，C不变
        if (self._op_type == UpSamplingOpType.BiLinearInterpolation):
            return (None, self._output_shape[0], self._output_shape[1], self._output_shape[2])
        #    reshape，改变H,W, C缩小
        elif (self._op_type == UpSamplingOpType.Reshape):
            return (None, H * S[0], W * S[1], C / (S[0] * S[1]))
        #    转置卷积，改变H,W, C不变
        elif (self._op_type == UpSamplingOpType.TransposedConv):
            return (None, H * 2, W * 2, C)
        #    填充，改变H,W, C不变
        elif (self._op_type == UpSamplingOpType.Filling):
            return (None, H * S[0], W * S[1], C)
        
        return (None, self._output_shape[0], self._output_shape[1], self._output_shape[2])
    
    #    reshape
    def reshape(self, x):
        return self._reshape(x)
    
    #    双线性插值
    def bi_linear_interpolation(self, x):
        B = x.shape[0]
        if (B == None): B = conf.DATASET_CELLS.get_batch_size()
        
        boxes = tf.repeat(tf.convert_to_tensor([[0,0, 1,1]], dtype=tf.float32), repeats=B, axis=0)
        boxes_idx = tf.range(B)
        y = tf.image.crop_and_resize(x, 
                                     boxes=boxes, 
                                     box_indices=boxes_idx, 
                                     crop_size=[self._output_shape[0], self._output_shape[1]])
        return y
        
    #    转置卷积
    def transposed_conv(self, x):
        return self._transpose_conv(x)
    
    #    填充
    def filling(self, x):
        S = self._strides
        x = tf.repeat(x, repeats=S[0], axis=1)
        x = tf.repeat(x, repeats=S[1], axis=2)
        return x
    
    def call(self, x, **kwargs):
        #    验证输入尺寸
        if (self._input_shape
                and self._input_shape != x.shape[1:]):
            raise Exception(self.name + " input_shape:" + str(self._input_shape) + " not equal x:" + str(x.shape))
        
        #    双线性插值
        if (self._op_type == UpSamplingOpType.BiLinearInterpolation):
            y = self.bi_linear_interpolation(x)
        #    reshape
        elif (self._op_type == UpSamplingOpType.Reshape):
            y = self.reshape(x)
        #    转置卷积
        elif (self._op_type == UpSamplingOpType.TransposedConv):
            y = self.transposed_conv(x)
        #    填充
        elif (self._op_type == UpSamplingOpType.Filling):
            y = self.filling(x)
        else:
            y = x
        
        #    验证输出尺寸
        if (self._output_shape
                and self._output_shape != y.shape[1:]):
            raise Exception(self.name + " output_shape:" + str(self._output_shape) + " not equal y:" + str(y.shape))
        
        return y
    pass


#    Yolo Hard输出层
class YoloHard(tf.keras.layers.Layer):
    '''
        输入: [H * W * C1]
        Conv: [3*3*C2] in=输入 strides=1 padding=1 norm=BN active=Mish out=[H * W * C2]
        Conv: [1*1*(num_anchors * num_classes + 5)] strides=1 padding=0 out=[H * W * (num_anchors*(num_classes+5))]
        Reshape: out=[H, W, num_anchors, num_classes+5]
    '''
    def __init__(self, 
                 name='YoloHard',
                 filters=16,
                 num_anchors=3,
                 num_classes=32,
                 input_shape=None,
                 output_shape=None,
                 **kwargs):
        '''
            @param filters: 3*3卷积核的通道数
            @param num_anchors: 每个cell中的anchor数
            @param num_classes: 总分类个数
            @param input_shape: 输入格式
        '''
        super(YoloHard, self).__init__(name=name, **kwargs)
        
        self._filters = filters
        self._num_anchors = num_anchors
        self._num_classes = num_classes
        
        self._input_shape = input_shape
        self._output_shape = output_shape
        
        out_filters = num_anchors * (num_classes + 5)
        #    输出格式为[H, W, num_anchors, num_classes+5]
        out_shape = (input_shape[0], input_shape[1], num_anchors, num_classes + 5)
        #    初始化网络结构
        self._layer = tf.keras.models.Sequential([
                #    3*3*filters卷积
                Conv2DNormActive(filters=filters, 
                                 kernel_size=[3, 3], 
                                 strides=1, 
                                 padding='SAME', 
                                 name=name + '_conv33'),
                #    1*1*(num_anchors * num_classes + 5)卷积
                tf.keras.layers.Conv2D(filters=out_filters, 
                                       kernel_size=[1, 1], 
                                       strides=1, 
                                       padding='VALID', 
                                       kernel_initializer=tf.initializers.he_normal(),
                                       bias_initializer=tf.initializers.zeros(),
                                       name=name + '_conv11'),
                tf.keras.layers.Reshape(target_shape=out_shape)
            ], name=name + '_layer')
        pass
    
    def call(self, x, **kwargs):
        #    验证输入尺寸
        if (self._input_shape
                and self._input_shape != x.shape[1:]):
            raise Exception(self.name + " input_shape:" + str(self._input_shape) + " not equal x:" + str(x.shape))
        
        #    过卷积层，拿到最终输出(batch_size, H, W, num_anchors, num_classes + 5)
        y = self._layer(x)
        
        #    y的输出格式(batch_size, H, W, num_anchors, num_classes + 5)
        #    (batch_size, H, W, num_anchors, num_classes)为anchor从属于每个分类的概率，该值过Sigmoid
        [y_cls, y_confidence, y_box_xy, y_box_wh] = tf.split(y, num_or_size_splits=[self._num_classes, 1, 2,2], axis=-1)
        #    y_cls过sigmoid
        y_cls = tf.nn.sigmoid(y_cls)
        #    y_confidence过sigmoid
        y_confidence = tf.nn.sigmoid(y_confidence)
        #    y_box_xy过sigmoid
        y_box_xy = tf.nn.sigmoid(y_box_xy)
        #    合并结果
        y = tf.concat([y_cls, y_confidence, y_box_xy, y_box_wh], axis=-1)
        
        #    验证输出尺寸
        if (self._output_shape
                and self._output_shape != y.shape[1:]):
            raise Exception(self.name + " output_shape:" + str(self._output_shape) + " not equal y:" + str(y.shape))
        return y
    pass


#    YoloHardRegister
class YoloHardRegister():
    _instance_lock = threading.Lock()
    
    '''YoloHard寄存器'''
    def __init__(self):
        pass
    
    @classmethod
    def instance(cls, *args, **kwargs):
        with YoloHardRegister._instance_lock:
            if not hasattr(YoloHardRegister, '_instance'):
                YoloHardRegister._instance = YoloHardRegister(*args, **kwargs)
            pass
        return YoloHardRegister._instance
    
    #    暂存yolohard1结果
    def deposit_yolohard1(self, y):
        self._yolohard1 = y
        pass
    def get_yolohard1(self):
        return self._yolohard1
    
    #    暂存yolohard2结果
    def deposit_yolohard2(self, y):
        self._yolohard2 = y
        pass
    def get_yolohard2(self):
        return self._yolohard2
    
    #    暂存yolohard3结果
    def deposit_yolohard3(self, y):
        self._yolohard3 = y
        pass
    def get_yolohard3(self):
        return self._yolohard3
    pass
#    YoloHardRegisterLayer
class YoloHardRegisterLayer(tf.keras.layers.Layer):
    def __init__(self,
                 name='YoloHardRegisterLayer',
                 yolohard_register=YoloHardRegister.instance(),
                 **kwargs):
        super(YoloHardRegisterLayer, self).__init__(name=name, trainable=False, **kwargs)
        self._yolohard_register = yolohard_register
        
        pass
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, x=None, yolohard1=None, yolohard2=None, yolohard3=None, **kwargs):
        self._yolohard_register.deposit_yolohard1(yolohard1)
        self._yolohard_register.deposit_yolohard2(yolohard2)
        self._yolohard_register.deposit_yolohard3(yolohard3)
        return x
    pass


#    最多跑一次原则
#    暂存yolohard解析出的liable_anchors，liable_num_objects，unliable_anchors, unliable_num_objects
class AnchorsRegister():
    '''暂存yolohard解析出来的结果。“最多跑一次”
        在一个batch中，takeout_liable_cells, takeout_liable_anchors, takeout_unliable_anchors的只跑一次
    '''
    _instance_lock = threading.Lock()
    
    def __init__(self):
        pass
    
    @classmethod
    def instance(cls, *args, **kwargs):
        with AnchorsRegister._instance_lock:
            if not hasattr(AnchorsRegister, '_instance'):
                AnchorsRegister._instance = AnchorsRegister(*args, **kwargs)
            pass
        return AnchorsRegister._instance
    
    
    #    暂存yolohard1解析结果
    def deposit_yolohard1(self, liable_anchors1, liable_num_objects1, unliable_anchors1, unliable_num_objects1):
        self._liable_anchors1 = liable_anchors1
        self._liable_num_objects1 = liable_num_objects1
        self._unliable_anchors1 = unliable_anchors1
        self._unliable_num_objects1 = unliable_num_objects1
        pass
    #    取yolohard1的解析结果
    def get_yolohard1(self):
        return self._liable_anchors1, self._liable_num_objects1, self._unliable_anchors1, self._unliable_num_objects1
    
    #    暂存yolohard2解析结果
    def deposit_yolohard2(self, liable_anchors2, liable_num_objects2, unliable_anchors2, unliable_num_objects2):
        self._liable_anchors2 = liable_anchors2
        self._liable_num_objects2 = liable_num_objects2
        self._unliable_anchors2 = unliable_anchors2
        self._unliable_num_objects2 = unliable_num_objects2
        pass
    #    取yolohard2的解析结果
    def get_yolohard2(self):
        return self._liable_anchors2, self._liable_num_objects2, self._unliable_anchors2, self._unliable_num_objects2
    
    #    暂存yolohard3解析结果
    def deposit_yolohard3(self, liable_anchors3, liable_num_objects3, unliable_anchors3, unliable_num_objects3):
        self._liable_anchors3 = liable_anchors3
        self._liable_num_objects3 = liable_num_objects3
        self._unliable_anchors3 = unliable_anchors3
        self._unliable_num_objects3 = unliable_num_objects3
        pass
    #    取yolohard2的解析结果
    def get_yolohard3(self):
        return self._liable_anchors3, self._liable_num_objects3, self._unliable_anchors3, self._unliable_num_objects3
    pass

