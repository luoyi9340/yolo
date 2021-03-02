# -*- coding: utf-8 -*-  
'''
k-means算法计算所有标注框中最具代表性的N个标注框尺寸
    - 所有图片、标注信息均统一缩放到指定比例
    - 所有输出的尺寸均为相对统一尺寸输出

@author: luoyi
Created on 2021年2月22日
'''
import numpy as np
import random
np.set_printoptions(suppress=True, threshold=16)

import utils.conf as conf
from utils.iou import iou_n21_np


#    anchor尺寸生成器
class AnchorCreator():
    def __init__(self, 
                 k=9, 
                 standard_scale=[conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT],
                 threshold_mind=0.001,
                 show_log=False,
                 ):
        '''
            @param k: k-means++算法最终输出多少个聚类
            @param standard_scale: 标准尺寸（所有的图像都按这个尺寸缩放）
            @param threshold_mind: 新老质心最小差异。若所有聚类的新老质点都<该阈值，则算法结束
        '''
        self._k = k
        self._standard_scale = standard_scale
        self._threshold_mind = threshold_mind
        self._show_log = show_log
        pass
    
    #    选举出k个最具代性的尺寸
    def election(self, 
                 data_iterator=None,
                 num_sample=1000):
        '''选举出k个最具代性的尺寸（会对原因进行统一化缩放）
            @param num_sample: 参与k-means聚类的数据量。label_iterator中能拿多少算多少
            @param data_iterator: 数据迭代器
        '''
        #    从迭代器中取样本
        samples = self._samples(data_iterator=data_iterator, num_sample=num_sample)
        #    如果samples的维度是4维，则计算面积。追加到第5维
        if (np.ndim(samples) < 5):
            areas = np.abs(samples[:,2] - samples[:,0]) * np.abs(samples[:,3] - samples[:,1])
            areas = np.expand_dims(areas, axis=-1)
            samples = np.concatenate([samples, areas], axis=-1)
            pass
        
        centroids = self._kpp_centroids(samples, k=self._k)
        centroids = self._kmeans(samples=samples, centroids=centroids, k=self._k, threshold_mind=self._threshold_mind)
        return centroids
    
    #    按照面积排序，并按规格切分，并计算wh
    def sort_split_hw(self, new_shape=[3, 3], centroids=None):
        '''
            @param new_shape: 切分的新shape
            @param centroids: 已经求出的质心
            @return: hw numpy(num, 2(h,w))
        '''
        #    质心按面积升序
        idx = np.argsort(centroids[:,4], axis=-1)
        centroids = centroids[idx]
        #    计算每个质心的宽高
        w = np.abs(centroids[:,2] - centroids[:,0])
        h = np.abs(centroids[:,3] - centroids[:,1])
        hw = np.concatenate([np.expand_dims(h, axis=-1), np.expand_dims(w, axis=-1)], axis=-1)
        #    按新的shape切分
        new_shape = [new_shape[0], new_shape[1], 2]
        hw = np.reshape(hw, newshape=new_shape)
        return hw
    
    #    选举出k个最具代性的尺寸
    def election_samples(self,
                         samples=None,
                         ):
        #    如果samples的维度是4维，则计算面积
        if (np.ndim(samples) < 5):
            areas = np.abs(samples[:,2] - samples[:,0]) * np.abs(samples[:,3] - samples[:,1])
            areas = np.expand_dims(areas, axis=-1)
            samples = np.concatenate([samples, areas], axis=-1)
            pass
        #    随机初始质心
#         centroids = self._random_centroid(samples=samples, k=self._k)
        centroids = self._kpp_centroids(samples, k=self._k)
        centroids = self._kmeans(samples=samples, centroids=centroids, k=self._k, threshold_mind=self._threshold_mind)
        return centroids
        pass
    
    #    通过给定的质心指定k_means迭代
    def _kmeans(self, samples=None, centroids=None, k=9, threshold_mind=0.0001):
        '''
            @param samples: 样本
            @param centroids: 质心
            @param threshold_mind: 最小距离阈值。小于此值的距离认为没有距离
        '''
        #    计算样本与当前所有质心的距离
        d = self._distance(samples, centroids)
        #    新老的质心
        old_centroids = centroids
        if (self._show_log): print('init centroids:', old_centroids)
        new_centroids = self._reflush_centroids(samples=samples, ious=d)
        if (self._show_log): print('new centroids', new_centroids)
        while (not self._is_finish(new_centroids=new_centroids, old_centroids=old_centroids, threshold_mind=threshold_mind)):
            d = self._distance(samples, new_centroids)
            old_centroids = new_centroids
            new_centroids = self._reflush_centroids(samples=samples, ious=d)
            if (self._show_log): print('new centroids', new_centroids)
            pass
        return new_centroids
    
    #    检测新老质心之间的距离是否都小于等于阈值
    def _is_finish(self, new_centroids=None, old_centroids=None, threshold_mind=0.0001):
        '''
            @param new_centroids: 新质心(k, 5)
            @param old_centroids: 老质心(k, 5)
            @param threshold_mind: 距离最小阈值
        '''
        #    计算新老质心之间的距离
        ds = np.abs(new_centroids[:,:4] - old_centroids[:,:4])
        for d in ds:
            if (d[0] > threshold_mind 
                or d[1] > threshold_mind 
                or d[2] > threshold_mind 
                or d[3] > threshold_mind):
                return False
            pass
        return True
    
    #    计算质心
    def _reflush_centroids(self, samples=None, ious=None):
        '''
            @param samples: 样本numpy(num, 5)
            @param ious: 每个样本与质心的IoU(num, k)
            @return 新的质心(k, 5)
        '''
        new_centroids = []
        #    每个样本的聚类(num,)
        classify = np.argmin(ious, axis=-1)
        for c in range(self._k):
            s = samples[classify == c]
            xl = np.mean(s[:,0])
            yl = np.mean(s[:,1])
            xr = np.mean(s[:,2])
            yr = np.mean(s[:,3])
            area = np.abs(xr - xl) * np.abs(yr - yl)
            new_centroids.append([xl,yl, xr,yr, area])
            pass
        return np.array(new_centroids)
    
    #    距离定义
    def _distance(self, samples=None, centroids=None):
        ious = self._iou(samples, centroids)
        return 1 - ious
    
    #    计算IoU
    def _iou(self, samples=None, centroids=None):
        '''
            @param samples: 样本[num, 5]
            @param centroids: 质心[k, 5]
            @return numpy(num, k)
                        每行表示每个样本对所有聚类中心的IoU，顺序与samples给出的一致
                        num: 样本个数
                        k: 聚类个数
        '''
        ious = []
        for centroid in centroids:
            #    求每个样本框与质心框交集面积
            iou = iou_n21_np(rect_srcs=samples, rect_tag=centroid)
            ious.append(iou)
            pass
        return np.stack(ious, axis=-1)
    
    #    随机初始质心
    def _random_centroid(self, samples=None, k=9):
        '''
            @param samples: 样本
            @param k: 聚类数
        '''
        min_x = np.min(samples[:,0])
        min_y = np.min(samples[:,1])
        max_x = np.max(samples[:,2])
        max_y = np.max(samples[:,3])
        #    在min_x ~ max_x之间随机生成2*k个数，min_y ~ max_y之间随机生成2*k个数
        x1, x2 = np.array([random.uniform(min_x, max_x) for _ in range(k)]), np.array([random.uniform(min_x, max_x) for _ in range(k)])
        x = np.stack([x1, x2], axis=0)
        xl = np.min(x, axis=0)
        xr = np.max(x, axis=0)
        y1, y2 = np.array([random.uniform(min_y, max_y) for _ in range(k)]), np.array([random.uniform(min_y, max_y) for _ in range(k)])
        y = np.stack([y1, y2], axis=0)
        yl = np.min(y, axis=0)
        yr = np.max(y, axis=0)
        #    numpy(num, 4)[xl,yl, xr,yr]
        centroids = np.stack([xl,yl, xr,yr], axis=0)
        centroids = centroids.T
        #    计算面积(num, 5)[xl,yl, xr,yr, area]
        areas = np.abs(centroids[:,2] - centroids[:,0]) * np.abs(centroids[:,3] - centroids[:,1])
        areas = np.expand_dims(areas, axis=-1)
        return np.concatenate([centroids, areas], axis=-1)
    
    #    从样本中初始质心(k-means++算法)
    def _kpp_centroids(self, samples=None, k=9):
        centroids = []
        samples_idx = np.arange(samples.shape[0])
        
        #    随机取1个样本点作为第1个质心
        frist_idx = np.random.randint(0, len(samples))
        centroid = samples[frist_idx]
        centroids.append(centroid)
        #    选为质心后从样本中删除（存在再次选到该样本的可能性。虽然很小。。。）
        np.delete(samples, [frist_idx], axis=0)
        np.delete(samples_idx, [frist_idx], axis=0)
        
        for _ in range(k - 1):
            #    计算所有样本与当前最近质心的距离
            crt_centroids = np.array(centroids)
            ious = self._distance(samples, crt_centroids)
            ious = np.min(ious, axis=-1)
            #    转换成每个样本被选择的概率
            sum_ious = np.sum(ious)
            p = ious / sum_ious
            #    轮盘赌
            centroid_idx = np.random.choice(samples_idx, size=1, p=p)
            centroids.append(np.squeeze(samples[centroid_idx]))
            #    选为质心后从样本中删除（存在再次选到该样本的可能性。虽然很小。。。）
            np.delete(samples, [centroid_idx], axis=0)
            np.delete(samples_idx, [centroid_idx], axis=0)
            pass
        
        return np.array(centroids)
    
    #    从迭代器中取样本
    def _samples(self, data_iterator=None, num_sample=1000):
        '''
            @param num_sample: 参与k-means聚类的数据量。label_iterator中能拿多少算多少
            @param data_iterator: 数据迭代器
            @return numpy(num, 4)
                            [lx,ly, rx,ry]
        '''
        crt_count = 0
        samples = []
        for _, _, _, _, labels in data_iterator:
            crt_count += 1
            if (crt_count > num_sample): break
            
            for label in labels:
                #    取左上点坐标
                lx = label[1]
                ly = label[2]
                #    取宽高，并换算成右下点坐标
                rx = lx + label[3]
                ry = ly + label[4]
                #    按比例缩放
                lx = lx
                ly = ly
                rx = rx
                ry = ry
                #    记录box
                samples.append([lx,ly, rx,ry])
                pass
            pass
        samples = np.array(samples)
        return samples
    pass



