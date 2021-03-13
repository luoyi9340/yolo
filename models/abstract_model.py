# -*- coding: utf-8 -*-  
'''
定义模型的公共行为

Created on 2020年12月16日

@author: irenebritney
'''
import abc
import tensorflow as tf

import utils.conf as conf
import utils.alphabet as alphabet 
from models.layer.commons.part import YoloHardRegister, AnchorsRegister
from models.layer.commons.preporcess import takeout_liables, takeout_unliables, parse_idxBHW
from models.layer.commons.metrics import YoloV4MetricsBoxes, YoloV4MetricsConfidence, YoloV4MetricsUnConfidence, YoloV4MetricsClasses
from models.layer.commons.nms import nms_tf


#    模型类的公共行为
class AModel(tf.keras.models.Model):
    '''模型的公共行为
        @method
        - 测试
        - 保存/载入模型参数
        - 训练
        @abstractmethod
        - 梯度下降方式
        - 损失函数
        - 评价标准
        - 装备模型
    '''
    def __init__(self, learning_rate=0.01, name="Model", auto_assembling=True, **kwargs):
        super(AModel, self).__init__(name=name, **kwargs)
        
        self._name = name
        self.__learning_rate=learning_rate
        
        if (auto_assembling):
            #    装配网络模型
            self.assembling()
            #    初始化优化器，损失，评价
            optimizer = self.create_optimizer(learning_rate=learning_rate)
            loss = self.create_loss()
            metrics = self.create_metrics()
            #    编译网络
            self.compile(optimizer=optimizer, 
                         loss=loss, 
                         metrics=metrics)
            pass
        pass
    
    
    #    保存模型参数
    def save_model_weights(self, filepath):
        '''保存模型参数
            @param filepath: 保存文件路径（建议文件以.ckpt为后缀）
        '''
        self.save_weights(filepath, overwrite=True, save_format="h5")
        pass
    
    #    加载模型参数
    def load_model_weight(self, filepath):
        '''加载模型参数
            @param filepath: 加载模型路径
        '''
        self.load_weights(filepath)
        pass
    
    
    #    训练模型（用tensor_db做数据源）
    def train_tensor_db(self, db_train=None, 
                        db_val=None,
                        steps_per_epoch=100,
                        batch_size=32, 
                        epochs=5,
                        auto_save_weights_after_traind=True,
                        auto_save_weights_dir=None,
                        auto_learning_rate_schedule=True,
                        auto_tensorboard=True,
                        auto_tensorboard_dir=None):
        '''训练模型
            @param db_train: 训练集
            @param db_val: 验证集
            @param val_split: 验证集占比，与db_val二选一
            @param batch_size: 批量喂数据大小
            @param epochs: epoch次数
            @param auto_save_weights_after_traind: 是否在训练完成后自动保存（默认True）
            @param auto_save_file_path: 当auto_save_epoch为true时生效，保存参数文件path
            @param auto_learning_rate_schedule: 是否动态调整学习率
            @param auto_tensorboard: 是否开启tensorboard监听（一款tensorflow自带的可视化训练过程工具）
            @param auto_tensorboard_dir: tensorboard日志写入目录
            @return: history
        '''
        #    初始化模型的各种回调
        callbacks = self.callbacks(auto_save_weights_after_traind, auto_save_weights_dir, 
                                   auto_learning_rate_schedule, 
                                   auto_tensorboard, auto_tensorboard_dir,
                                   batch_size=batch_size)
        
        his = self.fit_generator(db_train, 
                                 validation_data=db_val,
                                 steps_per_epoch=steps_per_epoch, 
                                 epochs=epochs, 
                                 verbose=1, 
                                 callbacks=callbacks,
                                 shuffle=False)
        return his
    #    训练模型
    def train(self, X_train, Y_train,
                    X_val, Y_val,
                    batch_size=32, 
                    epochs=5,
                    auto_save_weights_after_traind=True,
                    auto_save_weights_dir=None,
                    auto_learning_rate_schedule=True,
                    auto_tensorboard=True,
                    auto_tensorboard_dir=None
                    ):
        '''训练模型
            @param X_train: 训练集
            @param Y_train: 训练集标签
            @param X_val: 验证集（若验证集X，Y有一个为空或X，Y数量不对等则放弃验证）
            @param Y_val: 验证集标签（若验证集X，Y有一个为空或X，Y数量不对等则放弃验证）
            @param batch_size: 批量喂数据大小
            @param epochs: epoch次数
            @param auto_save_weights_after_traind: 是否在训练完成后自动保存（默认True）
            @param auto_save_file_path: 当auto_save_epoch为true时生效，保存参数文件path
            @param auto_learning_rate_schedule: 是否动态调整学习率
            @param auto_tensorboard: 是否开启tensorboard监听（一款tensorflow自带的可视化训练过程工具）
            @param auto_tensorboard_dir: tensorboard日志写入目录
            @return: history
        '''
        #    初始化模型的各种回调
        callbacks = self.callbacks(auto_save_weights_after_traind, auto_save_weights_dir, 
                                   auto_learning_rate_schedule, 
                                   auto_tensorboard, auto_tensorboard_dir, batch_size=batch_size)
        
        his = self.fit(x=X_train, y=Y_train,
                       batch_size=batch_size, 
                       epochs=epochs, 
                       verbose=1, 
                       validation_data=(X_val, Y_val),
                       callbacks=callbacks,
                       shuffle=False)
        return his

    #    模型回调
    def callbacks(self, auto_save_weights_after_traind=True, auto_save_weights_dir=None, 
                        auto_learning_rate_schedule=True,
                        auto_tensorboard=True,
                        auto_tensorboard_dir=None,
                        batch_size=128
                        ):
        '''初始化各种回调
            @param auto_save_weights_after_traind: 是否在每次epoch完成时保存模型
            @param auto_save_file_path: 保存模型路径(**/*,h5)
            @param auto_learning_rate_schedule: 是否开启自动更新学习率
            @param auto_tensorboard: 是否开启tensorboard
            @param auto_tensorboard_dir: tensorboard日志保存目录
            @param batch_size: tensorboard多少个数据后刷新
        '''
        #    训练期间的回调
        callbacks = []
        #    如果需要每个epoch保存模型参数
        if (auto_save_weights_after_traind):
            auto_save_file_path = auto_save_weights_dir + "/" + self.model_name() + '_{epoch:02d}_{val_loss:.2f}' + ".h5"
            conf.mkfiledir_ifnot_exises(auto_save_file_path)
            auto_save_weights_callback = tf.keras.callbacks.ModelCheckpoint(filepath=auto_save_file_path,
                                                                            monitor="val_loss",         #    需要监视的值
                                                                            verbose=1,                      #    信息展示模式，0或1
                                                                            save_best_only=False,           #    当设置为True时，将只保存在验证集上性能最好的模型，一般我们都会设置为True. 
                                                                            model='auto',                   #    ‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，
                                                                                                            #    例如:
                                                                                                            #        当监测值为val_acc时，模式应为max，
                                                                                                            #        当检测值为val_loss时，模式应为min。
                                                                                                            #        在auto模式下，评价准则由被监测值的名字自动推断。 
                                                                            save_weights_only=True,         #    若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等
                                                                            period=1                        #    CheckPoint之间的间隔的epoch数
                                                                            )
            callbacks.append(auto_save_weights_callback)
            
            #    保存参数时一并保存此时的配置项
            current_conf_file_path = auto_save_weights_dir + "/conf_" + self.model_name() + ".yml"
            conf.write_conf(conf.ALL_DICT, current_conf_file_path)
            pass
        #    如果需要在训练期间动态调整学习率
        if (auto_learning_rate_schedule):
#             lrCallback = AdjustLRWithBatchCallback(n_batch=200, m_batch=400)
#             callbacks.append(lrCallback)
            reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                        factor=0.1,             #    每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
                                                                        patience=1,             #    当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
                                                                        mode='auto',            #    ‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少
                                                                        epsilon=0.000000001,    #    阈值，用来确定是否进入检测值的“平原区” 
                                                                        cooldown=0,             #    学习率减少后，会经过cooldown个epoch才重新进行正常操作
                                                                        min_lr=0                #    学习率的下限（下不封顶）
                                                                        )
            callbacks.append(reduce_lr_on_plateau)
            pass
        #    如果需要在训练过程中开启tensorboard监听
        if (auto_tensorboard):
            #    tensorboard目录：tensorboard根目录 + / 模型名称_b{batch_size}_lr{learning_rate}
            tensorboard_dir = auto_tensorboard_dir + "/" + self.model_name() + "_b" + str(batch_size) + "_lr" + str(self.__learning_rate)
            conf.mkdir_ifnot_exises(tensorboard_dir)
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir,               #    tensorboard主目录
                                                         histogram_freq=1,                      #    对于模型中各个层计算激活值和模型权重直方图的频率（训练轮数中）。 
                                                                                                #        如果设置成 0 ，直方图不会被计算。对于直方图可视化的验证数据（或分离数据）一定要明确的指出。
                                                         write_graph=True,                      #    是否在 TensorBoard 中可视化图像。 如果 write_graph 被设置为 True
                                                         write_grads=True,                      #    是否在 TensorBoard 中可视化梯度值直方图。 
                                                                                                #        histogram_freq 必须要大于 0
                                                         batch_size=batch_size,                 #    用以直方图计算的传入神经元网络输入批的大小
                                                         write_images=True,                     #    是否在 TensorBoard 中将模型权重以图片可视化，如果设置为True，日志文件会变得非常大
                                                         embeddings_freq=None,                  #    被选中的嵌入层会被保存的频率（在训练轮中）
                                                         embeddings_layer_names=None,           #    一个列表，会被监测层的名字。 如果是 None 或空列表，那么所有的嵌入层都会被监测。
                                                         embeddings_metadata=None,              #    一个字典，对应层的名字到保存有这个嵌入层元数据文件的名字
                                                         embeddings_data=None,                  #    要嵌入在 embeddings_layer_names 指定的层的数据。 Numpy 数组（如果模型有单个输入）或 Numpy 数组列表（如果模型有多个输入）
                                                         update_freq='batch'                    #    'batch' 或 'epoch' 或 整数。
                                                                                                #        当使用 'batch' 时，在每个 batch 之后将损失和评估值写入到 TensorBoard 中。
                                                                                                #        同样的情况应用到 'epoch' 中。
                                                                                                #        如果使用整数，例如 10000，这个回调会在每 10000 个样本之后将损失和评估值写入到 TensorBoard 中。注意，频繁地写入到 TensorBoard 会减缓你的训练。
                                                         )
            callbacks.append(tensorboard)
            pass
        return callbacks

    #    打印模型信息
    def show_info(self):
        self.summary()
        pass

    '''以下是抽象方法定义
        python毕竟不是Java，找不到更多约束了。。。
    '''
    #    子类必须指明梯度更新方式
    @abc.abstractclassmethod
    def create_optimizer(self, learning_rate=0.9):
        raise Exception('subclass must implement.')
    #    子类必须指明损失函数
    @abc.abstractclassmethod
    def create_loss(self):
        raise Exception('subclass must implement.')
    #    子类必须指明评价方式
    @abc.abstractclassmethod
    def create_metrics(self):
        raise Exception('subclass must implement.')
    #    装配模型
    @abc.abstractclassmethod
    def assembling(self, net):
        raise Exception('subclass must implement.')
    
    #    前向传播
    def call(self, inputs, training=None, mask=None):
        return tf.keras.models.Model.call(self, inputs, training=training, mask=mask)
    
    #    模型名称
    def model_name(self):
        return self.name
    
    
    #    计算yolohard的评价
    def metrics_yolohard(self, y_true, total, yolohard, idx_yolohard,
                         anchors_register=AnchorsRegister.instance(),
                         num_anchors=conf.DATASET_CELLS.get_anchors_set().shape[1],
                         num_classes=len(alphabet.ALPHABET),
                         threshold_liable_iou=conf.V4.get_threshold_liable_iou(),
                         metrics_boxes = YoloV4MetricsBoxes(),
                         metrics_confidence = YoloV4MetricsConfidence(),
                         metrics_unconfidence = YoloV4MetricsUnConfidence(),
                         metrics_classes = YoloV4MetricsClasses()):
        liable_idxBHW, liable_num_objects = parse_idxBHW(y_true, total)
        liable_anchors, liable_num_objects = takeout_liables(liable_idxBHW, liable_num_objects, yolohard, y_true, num_anchors, total, num_classes, threshold_liable_iou)
        unliable_anchors, unliable_num_objects = takeout_unliables(liable_idxBHW, liable_num_objects, yolohard, y_true, total, num_anchors, num_classes)
        if (idx_yolohard == 1): anchors_register.deposit_yolohard1(liable_anchors, liable_num_objects, unliable_anchors, unliable_num_objects)
        if (idx_yolohard == 2): anchors_register.deposit_yolohard2(liable_anchors, liable_num_objects, unliable_anchors, unliable_num_objects)
        if (idx_yolohard == 3): anchors_register.deposit_yolohard3(liable_anchors, liable_num_objects, unliable_anchors, unliable_num_objects)
        
        #    返回值
        r = {}
        
        #    计算各种评价
        metrics_boxes.set_yolohard_scale_idx(idx_yolohard)
        metrics_boxes.update_state(y_true=y_true, y_pred=None, sample_weight=None)
        r['mae_box'] = metrics_boxes.result().numpy()
            
        metrics_confidence.set_yolohard_scale_idx(idx_yolohard)
        metrics_confidence.update_state(y_true=y_true, y_pred=None, sample_weight=None)
        r['metrics_confidence'] = metrics_confidence.result().numpy()
            
        metrics_unconfidence.set_yolohard_scale_idx(idx_yolohard)
        metrics_unconfidence.update_state(y_true=y_true, y_pred=None, sample_weight=None)
        r['metrics_unconfidence'] = metrics_unconfidence.result().numpy()
            
        metrics_classes.set_yolohard_scale_idx(idx_yolohard)
        metrics_classes.update_state(y_true=y_true, y_pred=None, sample_weight=None)
        r['metrics_classes'] = metrics_classes.result().numpy()
        
        return r
    #    测试准确率
    def test_metrics(self, X_test, Y_test, 
                     yolohard_register=YoloHardRegister.instance(),
                     anchors_register=AnchorsRegister.instance(),
                     num_anchors=conf.DATASET_CELLS.get_anchors_set().shape[1],
                     num_classes=len(alphabet.ALPHABET),
                     batch_size=conf.DATASET_CELLS.get_batch_size(),
                     threshold_liable_iou=conf.V4.get_threshold_liable_iou()):
        '''跑测试集
            @param X_test: 测试集    x.shape = (total, conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT, 3)
            @param Y_test: 测试集标签 y.shape = (total, num_scales, max_objects, 2 + 5 + num_anchors * 3)
            @return: 准确率（0 ~ 1之间）
        '''
        total = X_test.shape[0]
        #    取输入的预测结果    Tensor()
        _ = self(inputs=X_test)
        
        #    计算评价的工具
        metrics_boxes = YoloV4MetricsBoxes()
        metrics_confidence = YoloV4MetricsConfidence()
        metrics_unconfidence = YoloV4MetricsUnConfidence()
        metrics_classes = YoloV4MetricsClasses()
        
        #    最终输出
        res = {}
        
        #    shape=(total, H, W, num_anchors, num_classes + 5)
        yolohard1 = yolohard_register.get_yolohard1()
        if (yolohard1 is not None):
            yolohard_shape = (yolohard1.shape[1], yolohard1.shape[2])
            y_true = Y_test[:, 0, :]
            r = self.metrics_yolohard(y_true, total, yolohard1, 1, anchors_register, num_anchors, num_classes, threshold_liable_iou, metrics_boxes, metrics_confidence, metrics_unconfidence, metrics_classes)
            res[yolohard_shape] = r
            pass
        yolohard2 = yolohard_register.get_yolohard2()
        if (yolohard2 is not None):
            yolohard_shape = (yolohard2.shape[1], yolohard2.shape[2])
            y_true = Y_test[:, 1, :]
            r = self.metrics_yolohard(y_true, total, yolohard2, 2, anchors_register, num_anchors, num_classes, threshold_liable_iou, metrics_boxes, metrics_confidence, metrics_unconfidence, metrics_classes)
            res[yolohard_shape] = r
            pass
        yolohard3 = yolohard_register.get_yolohard3()
        if (yolohard3 is not None):
            yolohard_shape = (yolohard3.shape[1], yolohard3.shape[2])
            y_true = Y_test[:, 2, :]
            r = self.metrics_yolohard(y_true, total, yolohard3, 3, anchors_register, num_anchors, num_classes, threshold_liable_iou, metrics_boxes, metrics_confidence, metrics_unconfidence, metrics_classes)
            res[yolohard_shape] = r
            pass        
        
        return res
    
    
    #    解析yolohard的预测结果
    def parse_yolohard(self, 
                       yolohard, 
                       img_shape=None,
                       anchor_scale=None,
                       scale=None,
                       num_classes=len(alphabet.ALPHABET), 
                       threshold_liable_iou=conf.V4.get_threshold_liable_iou()):
        '''
            @param anchor_set: Tensor(3, 2)    anchor高宽（相对于统一尺寸原图）
            @param yolohard: Tensor(1, H, W, num_anchors, num_classes + 5)
                                                                num_class: 各个分类预测得分
                                                                1: 置信度预测
                                                                4: 预测的dx,dy,dw,dh
            @return: Tensor(sum_qualified_anchors, 4 + 1 + 1)
                                4: lx,ly,rx,ry    (相对统一尺寸坐标)
                                1: 置信度
                                1: 预测分类索引
        '''
        #    取置信度超过阈值的anchor    Tensor(sum_qualified_anchors, num_classes + 5)
        qualified_indices = tf.where(yolohard[:, :,:, :, num_classes] > threshold_liable_iou)
        qualified_indices = tf.cast(qualified_indices, dtype=tf.int32)
        qualified_anchors = tf.gather_nd(yolohard, indices=qualified_indices)
        sum_qualified_anchors = qualified_anchors.shape[0]                          #    符合条件的anchor总数
        #    按置信度降序
        qualified_anchors_confidence = qualified_anchors[:, num_classes]            #    Tensor(sum_qualified_anchors, )
        sort_confidence_desc = tf.argsort(qualified_anchors_confidence, axis=-1, direction='DESCENDING')
        #    排好序的anchor       Tensor(sum_qualified_anchors, num_classes + 5)
        qualified_anchors = tf.gather(qualified_anchors, indices=sort_confidence_desc, axis=0)
        #    排好序的anchor索引    Tensor(sum_qualified_anchors, 4)
        qualified_indices = tf.gather(qualified_indices, indices=sort_confidence_desc, axis=0)
        
        #    anchor_scale转换为相对特征图的高宽    Tensor(3, 2)
        fmaps_shape = tf.convert_to_tensor([[yolohard.shape[1], yolohard.shape[2]]], dtype=tf.float64)                    #    特征图高宽 Tensor(1, 2)                                                #    特征图高宽
        anchor_scale = anchor_scale / tf.convert_to_tensor([[conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT]], dtype=tf.float64)    #    anchor_scale转换为整图占比 Tensor(3, 3)
        anchor_scale = anchor_scale * fmaps_shape                                                                         #    anchor_scale转换为相对特征图高宽
        
        #    给qualified_anchors追加anchor高宽属性
        anchor_scale = tf.repeat(tf.expand_dims(anchor_scale, axis=0), repeats=sum_qualified_anchors, axis=0)        #    Tensor(sum_qualified_anchors, 3, 2)
        idx_ = tf.range(sum_qualified_anchors)
        idx_ = tf.stack([idx_, qualified_indices[:, 3]], axis=-1)
        anchor_scale = tf.gather_nd(anchor_scale, idx_)                                                              #    Tensor(sum_qualified_anchors, 2)
        qualified_indices = tf.cast(qualified_indices, dtype=tf.float64)
        qualified_indices = tf.concat([qualified_indices, anchor_scale], axis=-1)
        
        #    还原anchor_box
        dn = qualified_anchors[:, num_classes + 1:]                       #    Tensor(sum_qualified_anchors, 6)
        dn = tf.cast(dn, dtype=tf.float64)
        cx = dn[:, 0] + qualified_indices[:, 2]                           #    cx = dx + cell[x]
        cy = dn[:, 1] + qualified_indices[:, 1]                           #    cy = dy + cell[y]
        half_w = tf.math.exp(dn[:, 2]) * (qualified_indices[:, 5]) / 2.   #    w = exp(dw) * anchor[w]
        half_h = tf.math.exp(dn[:, 3]) * (qualified_indices[:, 4]) / 2.   #    h = exp(dh) * anchor[h]
        lx = cx - half_w                                                  #    Tensor(sum_qualified_anchors, )
        ly = cy - half_h                                                  #    Tensor(sum_qualified_anchors, )
        rx = cx + half_w                                                  #    Tensor(sum_qualified_anchors, )
        ry = cy + half_h                                                  #    Tensor(sum_qualified_anchors, )
        scale_w = img_shape[1] / yolohard.shape[2]
        scale_h = img_shape[0] / yolohard.shape[1]
        #    还原为相对统一尺寸原图坐标
        lx = lx * scale_w
        ly = ly * scale_h
        rx = rx * scale_w
        ry = ry * scale_h
        
        #    取各个分类得分
        qualified_anchors_classes = tf.math.argmax(qualified_anchors[:, :num_classes], axis=-1)
        qualified_anchors_classes = tf.cast(qualified_anchors_classes, dtype=tf.float64)
        
        #    取置信度
        qualified_anchors_confidence = qualified_anchors[:, num_classes]
        qualified_anchors_confidence = tf.cast(qualified_anchors_confidence, dtype=tf.float64)
        
        #    anchors_boxes 相对于特征图的坐标
        anchors_boxes = tf.stack([lx, ly, rx, ry, 
                                  qualified_anchors_confidence,
                                  qualified_anchors_classes], axis=-1)
        return anchors_boxes
    #    预测一张图片的结果
    def divination(self, img,
                   yolohard_register=YoloHardRegister.instance(),
                   img_shape=None,
                   anchors_set=conf.DATASET_CELLS.get_anchors_set(),
                   scales_set=conf.DATASET_CELLS.get_scales_set(),
                   num_classes=len(alphabet.ALPHABET),
                   threshold_liable_iou=conf.V4.get_threshold_liable_iou(),
                   threshold_overlap_iou=conf.V4.get_threshold_overlap_iou()
                   ):
        ''' step1: 拿图片的预测特征图
            step2: 通过预测特征图拿置信度超过阈值的anchors，并按置信度降序
            step3: 置信度超过阈值的anchors还原出anchor_box
                        - 通过anchor预测拿dx,dy,dw,dh
                        - 通过anchor在cell中的索引拿anchor[w],anchor[h]
                        - 通过anchor所在cell在特征图的索引拿cell的左上点坐标
                        - 还原出的anchor_box是相对特征图的，还原为原图
                        - 还原出的原图是相对统一比例缩放的，这里不负责还原真实原图
            step4: anchor_box做非极大值抑制
                        - 置信度降序对应的anchor_box取第1个，记录进最终判定
                        - 计算第1个与其他box的IoU，超过阈值的判定为重复。从生下的anchor_box中过滤掉重复的box
                        - 在生下的anchor_box中取第1个，记录进最终判定，重复上述步骤直到没有anchor_box
            @param img: Tensor(conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT, 3)，值归一到[-1,1]之间
            @return: Tensor(sum_qualified_anchors, 6)
                        lx,ly, rx,ry, 置信度, idxV
                        坐标为相对统一比例缩放的原图，需要还原真实原图自行按比例缩放
        '''
        #    给img增加1维，按照(batch_size, H, W, 3)的格式给进网络
        img = tf.expand_dims(img, axis=0)
        
        #    过网络，拿特征图
        _ = self(inputs=img)
        #    yolohard.shape=(H, W, num_anchors, num_classes + 5)
        yolohard1 = yolohard_register.get_yolohard1()
        anchors_boxes_list = []
        if (yolohard1 is not None): 
            anchors_boxes = self.parse_yolohard(yolohard1, img_shape=img_shape, anchor_scale=tf.convert_to_tensor(anchors_set[0], dtype=tf.float64), scale=scales_set[0], num_classes=num_classes, threshold_liable_iou=threshold_liable_iou)
            anchors_boxes_list.append(anchors_boxes)
            pass
        yolohard2 = yolohard_register.get_yolohard2()
        if (yolohard2 is not None):
            anchors_boxes = self.parse_yolohard(yolohard2, img_shape=img_shape, anchor_scale=tf.convert_to_tensor(anchors_set[1], dtype=tf.float64), scale=scales_set[1], num_classes=num_classes, threshold_liable_iou=threshold_liable_iou)
            anchors_boxes_list.append(anchors_boxes)
            pass
        yolohard3 = yolohard_register.get_yolohard3()
        if (yolohard3 is not None):
            anchors_boxes = self.parse_yolohard(yolohard3, img_shape=img_shape, anchor_scale=tf.convert_to_tensor(anchors_set[2], dtype=tf.float64), scale=scales_set[2], num_classes=num_classes, threshold_liable_iou=threshold_liable_iou)
            anchors_boxes_list.append(anchors_boxes)
            pass
        anchors_boxes = tf.concat(anchors_boxes_list, axis=0)
        
        #    非极大值抑制
        anchors_boxes = tf.gather(anchors_boxes, indices=tf.argsort(anchors_boxes[:, 4], axis=-1, direction='DESCENDING'), axis=0)
        anchors_boxes = nms_tf(anchors_boxes, threshold_overlap_iou)
        
        #    结果按左上点x坐标排序
        anchors_boxes = tf.gather(anchors_boxes, indices=tf.argsort(anchors_boxes[:, 0], axis=-1), axis=0)
        
        return anchors_boxes
    
    
    pass



