# -*- coding: utf-8 -*-  
'''
日志工厂

Created on 2020年12月15日

@author: irenebritney
'''
import logging
import yaml

import utils.conf as conf

#    取配置文件目录
LOG_CONF_PATH = conf.ROOT_PATH + "/resources/logs.yml"
#    日志对象缓存
LOGS_CACHE = {}

#    取logger对象（与conf.yml中logs配置项对应）
def get_logger(log_name):
    log = LOGS_CACHE.get(log_name)
    if (log is None): 
        log = init_logger_by_name(log_name)
        LOGS_CACHE[log_name] = log
    return log
#    取tf.print需要的重定向文件目录
def get_logger_filepath(log_name):
    log = LOGS_CACHE.get(log_name)
    if (log is None): 
        log = init_logger_by_name(log_name)
        LOGS_CACHE[log_name] = log
    return "file://" + log.log_file_path
#    取rootlog（但不建议这样用）
def get_root():
    return get_logger('root')



#    日志级别与logging枚举相互转化
def log_level_enums(level="INFO"):
    level = level.upper()
    if (level == "NOTSET"): return logging.NOTSET
    elif (level == "DEBUG"): return logging.DEBUG
    elif (level == "INFO"): return logging.INFO
    elif (level == "WARNING" | level == "WARN"): return logging.WARNING
    elif (level == "ERROR"): return logging.ERROR
    elif (level == "FAIL" | level == "CRITICAL"): return logging.FATAL
    else: return logging.INFO
    
    
#    初始化控制台输出格式
def console_handler(log_fmt="%(asctime)s-%(name)s-%(levelname)s-%(message)s", log_level=logging.INFO):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    #    root log输出格式
    console_fmt = logging.Formatter(log_fmt)
    console_handler.setFormatter(console_fmt)
    return console_handler


#    初始化root_log
def init_root_logger():
    root_log = logging.getLogger("root");
    root_log.setLevel(logging.INFO)
    
    #    root log输出位置
    root_log_path = conf.ROOT_PATH + "/logs/root.log"
    conf.mkfiledir_ifnot_exises(root_log_path)
    root_log_handler = logging.FileHandler(root_log_path, encoding='utf-8')
    root_log_handler.setLevel(logging.INFO)
    root_log.log_file_path = root_log_path
    
    #    root log输出格式
    root_log_fmt = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s")
    root_log_handler.setFormatter(root_log_fmt)
    
    root_log.addHandler(root_log_handler)
    root_log.addHandler(console_handler())
    LOGS_CACHE['root'] = root_log
    return root_log
ROOT_LOG = init_root_logger()


#    初始化log
def init_logger_by_name(log_name):
    log_name_conf = LOGS_CONF.get(log_name)
    #    如果还没有配置值直接用root
    if (log_name_conf is None): 
        LOGS_CACHE[log_name] = ROOT_LOG
        return ROOT_LOG
    
    #    否则根据配置初始化
    log = logging.getLogger(log_name)
    
    log_level = log_level_enums(log_name_conf.get('level'))
    log.setLevel(log_level)
    
    log_path = log_name_conf.get('out')
    if (log_path is None): log_path = "logs/" + log_name + ".log"
    if (not log_path.startswith("/")): log_path = conf.ROOT_PATH + "/" + log_path
    conf.mkfiledir_ifnot_exises(log_path)
    log.log_file_path = log_path
    
    log_handler = logging.FileHandler(log_path, encoding='utf-8')
    log_handler.setLevel(log_level)
    
    log_fmt = log_name_conf.get('formatter')
    if (log_fmt is not None): 
        fmt = logging.Formatter(log_fmt)
        log_handler.setFormatter(fmt)
        pass
    
    log.addHandler(log_handler)
    
    is_console = log_name_conf.get('console')
    if (is_console): log.addHandler(console_handler(log_fmt=log_fmt, log_level=log_level))
    
    return log


#    加载logs.yml配置文件
def load_logs_yaml(yml_fpath=LOG_CONF_PATH):
    print('加载日志配置文件:' + yml_fpath)
    
    f = open(yml_fpath, 'r', encoding='utf-8')
    fr = f.read()
    
    logs = yaml.safe_load(fr)
    return logs
LOGS_CONF = load_logs_yaml()


