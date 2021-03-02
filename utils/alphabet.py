# -*- coding: utf-8 -*-  
'''
字母表

Created on 2020年12月15日

@author: luoyi
'''
#    字母表
ALPHABET = "3456789ABCDEFGHJKLMNPQRSTUVWXYZ"
alphabet_map = {}

#    类别<->编码
def category_index(category):
    '''类别转换为编码
    '''
    if (not alphabet_map):
        i = 0
        for c in ALPHABET:
            alphabet_map[c] = i
            i = i + 1
            pass
        pass
    return alphabet_map.get(category, -1)
#    编码<->类别
def index_category(index):
    '''编码转换类别
    '''
    return ALPHABET[index]



