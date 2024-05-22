import random
import numpy as np

data_path = 'D:/paper_projects/data/'

def set_seed(seed):
    random.seed(seed)  # 固定随机数种子

def hw_sample(h, w, num):
    '''取num个传感器

    :param int h: 场高度
    :param int w: 场宽度
    :param float rate: 比例
    :return list[tuple]: 采样后列表
    '''
    all_positions = [(i, j) for i in range(h) for j in range(w)]
    positions = random.sample(all_positions, num)

    return np.array(positions)

def ss_sample(ss, num):
    '''取num个传感器

    :param list ss: 传感器位置列表
    :param float rate: 比例
    :return list: 采样后列表
    '''
    num = min(num, len(ss))
    positions = random.sample(ss, num)
    
    return np.array(positions)

def ss_lost(ss, num):
    '''丢失num个传感器

    :param list ss: 传感器位置列表
    :param int num: 丢失数量
    :return list: 采样后列表
    '''
    positions = random.sample(ss, max(len(ss)-num, 1))
    
    return np.array(positions)

ss_easy = [[71, 76], [69, 175], [49, 138],
            [56, 41], [61, 141], [41, 30],
            [40, 177], [55, 80], [41, 60],
            [60, 70], [60, 100], [51, 120],
            [80, 160], [50, 165], [60, 180],
            [70, 30]] # 16个

ss_moderate = [[71, 76], [69, 175], [49, 138],
                [56, 41], [61, 141], [41, 30],
                [40, 177], [55, 80]] # 8个

ss_hard = [[41, 30], [56, 41], [55, 80], [71, 76]] # 8个

sensorset = {'easy': ss_easy, 'moderate': ss_moderate, 'hard': ss_hard}
