import torch
from torch.utils.data import Dataset

import pickle
import h5py
import numpy as np

from tqdm import tqdm
from scipy.interpolate import griddata

from .config import data_path, set_seed, hw_sample, ss_sample

class CylinderMask(Dataset):
    def __init__(self, index=[0], seed=1, num=4, mean=None, std=None, scale=2, steps=1):
        """
        圆柱绕流数据集：掩码嵌入
        
        :param index: 快照索引
        :param seed: 随机种子
        :param num: 传感器数量
        :param mean: 数据均值
        :param std: 数据标准差
        :param scale: 标准差缩放因子
        :param steps: 步长
        """
        super().__init__()
        set_seed(seed)
        self.steps = steps
        with open(f'{data_path}cylinder/Cy_Taira.pickle', 'rb') as df:
            data = pickle.load(df)
            data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)
            self.data = data[index]  # n, c, h, w

        mask = (self.data < -0.5) | (self.data > 0.5)
        self.mean = self.data[mask].mean() if mean is None else mean
        self.std = self.data[mask].std() * scale if std is None else std

        # 标准化数据
        self.data = (self.data - self.mean) / self.std

        # 使用高级索引创建稀疏数据张量
        self.size = self.data.shape[2:] # 后两位
        positions = hw_sample(*self.size, num)
        self.observe = torch.zeros_like(self.data)
        self.observe[:, :, positions[:, 0], positions[:, 1]] = self.data[:, :, positions[:, 0], positions[:, 1]]

    def __getitem__(self, index):
        if self.steps < 2:
            return self.observe[index, :], self.data[index, :]
        else:
            end_index = index + self.steps
            features = self.observe[index:end_index, :]

            return features, self.data[end_index-1, :]

    def __len__(self):
        return self.data.shape[0]-self.steps
    
    
class CylinderVoronoi(Dataset):
    def __init__(self, index=[0], seed=1, num=4, mean=None, std=None, scale=2, steps=1):
        """
        圆柱绕流数据集：voronoi嵌入
        
        :param index: 快照索引
        :param positions: 传感器位置(h, w)
        """
        super().__init__()
        set_seed(seed)
        self.steps = steps
        
        with open(f'{data_path}cylinder/Cy_Taira.pickle', 'rb') as df:
            data = pickle.load(df)
            data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)
            self.data = data[index]  # n, c, h, w

        mask = (self.data < -0.5) | (self.data > 0.5)
        self.mean = self.data[mask].mean() if mean is None else mean
        self.std = self.data[mask].std() * scale if std is None else std

        # 标准化数据，缩放的[-5, 5]
        self.data = (self.data - self.mean) / self.std

        h, w = self.data.shape[2:]
        self.size = (h, w)
        positions = hw_sample(h, w, num)
        
        x_coor, y_coor = np.linspace(0, w-1, w), np.linspace(0, h-1, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)

        # 使用高级索引提取数据
        sparse_data = self.data[:, 0, positions[:, 0], positions[:, 1]]
        
        # 最近邻插值，用方格坐标即可。实际位置中x-y方向等差均为0.08，与方格相同
        voronoilist = []
        for i in tqdm(range(len(index))):# 遍历快照索引
            voronoi = griddata(positions, sparse_data[i], (y_coor, x_coor), method='nearest')
            voronoilist.append(torch.from_numpy(voronoi).float())

        self.voronoi = torch.stack(voronoilist).unsqueeze(1) # 增加channel维度
        
        self.posmask = torch.zeros_like(self.voronoi)
        self.posmask[:, :, positions[:, 0], positions[:, 1]] = 1
        
        pass # 调试用

    def __getitem__(self, index):
        # index为整数，返回无批次维度数据
        if self.steps < 2:
            return torch.cat([self.voronoi[index, :], self.posmask[index, :]], dim=0), self.data[index, :]
        else:
            end_index = index + self.steps
            features = torch.cat([self.voronoi[index:end_index, :], self.posmask[index:end_index, :]], dim=1)

            return features, self.data[end_index-1, :]

    def __len__(self):
        return self.data.shape[0]-self.steps
    

class SSTMask(Dataset):
    def __init__(self, index=[0], seed=1, num=10, mean=None, std=None, scale=1, steps=1) -> None:
        super().__init__()
        set_seed(seed)
        self.steps = steps
        with h5py.File(f'{data_path}NOAA/sst_weekly.mat', 'r') as f:
            lat = np.array(f['lat'])
            lon = np.array(f['lon'])
            sst = np.array(f['sst']) # 温度有未测量点
            time = np.array(f['time'])
        
        sst = sst.reshape(-1, len(lat[0,:]), len(lon[0,:]), order='F')
        nan_indices = np.argwhere(np.isnan(sst[0]))
        non_nan_indices = np.argwhere(~np.isnan(sst[0])) # 第0帧避开nan位置
        self.size = sst[0].shape  # sst形状
        positions = ss_sample([tuple(coord) for coord in non_nan_indices], num)
        
        self.data = torch.from_numpy(np.nan_to_num(sst))[index,:]
        self.mean = mean if mean is not None else self.data[:,non_nan_indices[:,0],non_nan_indices[:,1]].mean()
        self.std = std if std is not None else self.data[:,non_nan_indices[:,0],non_nan_indices[:,1]].std() * scale

        # 标准化数据
        self.data = (self.data - self.mean) / self.std
        self.data[:,nan_indices[:,0],nan_indices[:,1]] = 0
        self.data.unsqueeze_(1) # 增加channel维度
        
        # 采样
        self.observe = torch.zeros_like(self.data)
        self.observe[:,:,positions[:,0],positions[:,1]] = self.data[:,:,positions[:,0],positions[:,1]]
        
    def __getitem__(self, index):
        if self.steps < 2:
            return self.observe[index, :], self.data[index, :]
        else:
            end_index = index + self.steps
            features = self.observe[index:end_index, :]

            return features, self.data[end_index-1, :]

    def __len__(self):
        return self.data.shape[0]-self.steps
    
    
class SSTVoronoi(Dataset):
    def __init__(self, index=[0], seed=1, num=10, mean=None, std=None, scale=1, steps=1) -> None:
        super().__init__()
        set_seed(seed) # 设置采样种子
        self.steps = steps
        with h5py.File(f'{data_path}NOAA/sst_weekly.mat', 'r') as f:
            lat = np.array(f['lat'])
            lon = np.array(f['lon'])
            sst = np.array(f['sst']) # 温度有未测量点
            time = np.array(f['time'])
        
        sst = sst.reshape(-1, len(lat[0,:]), len(lon[0,:]), order='F')
        nan_indices = np.argwhere(np.isnan(sst[0]))
        non_nan_indices = np.argwhere(~np.isnan(sst[0])) # 第0帧避开nan位置
        self.size = sst[0].shape  # sst形状
        positions = ss_sample([tuple(coord) for coord in non_nan_indices], num)
        # positions = non_nan_indices[np.random.choice(non_nan_indices.shape[0], num, replace=False)]
        
        self.data = torch.from_numpy(np.nan_to_num(sst))[index,:]
        self.mean = mean if mean is not None else self.data[:,non_nan_indices[:,0],non_nan_indices[:,1]].mean()
        self.std = std if std is not None else self.data[:,non_nan_indices[:,0],non_nan_indices[:,1]].std() * scale

        # 标准化数据
        self.data = (self.data - self.mean) / self.std
        self.data[:, nan_indices[:,0], nan_indices[:,1]] = 0
        self.data.unsqueeze_(1) # 增加channel维度
        
        _, _, h, w = self.data.shape # NCHW
        x_coor, y_coor = np.linspace(0, w-1, w), np.linspace(0, h-1, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)
        
        # 使用高级索引提取数据
        sparse_data = self.data[:, 0, positions[:, 0], positions[:, 1]]
        
        # 最近邻插值，用方格坐标即可。实际位置中x-y方向等差均为0.08，与方格相同
        voronoilist = []
        for i in tqdm(range(len(index))):# 遍历快照索引
            voronoi = griddata(positions, sparse_data[i], (y_coor, x_coor), method='nearest')
            voronoilist.append(torch.from_numpy(voronoi).float())

        self.voronoi = torch.stack(voronoilist).unsqueeze(1) # 增加channel维度
        self.voronoi[:, :, nan_indices[:,0], nan_indices[:,1]] = 0
        
        self.posmask = torch.zeros_like(self.voronoi)
        self.posmask[:, :, positions[:, 0], positions[:, 1]] = 1
        
    def __getitem__(self, index):
        # index为整数，返回无批次维度数据
        if self.steps < 2:
            return torch.cat([self.voronoi[index, :], self.posmask[index, :]], dim=0), self.data[index, :]
        else:
            end_index = index + self.steps
            features = torch.cat([self.voronoi[index:end_index, :], self.posmask[index:end_index, :]], dim=1)

            return features, self.data[end_index-1, :]

    def __len__(self):
        return self.data.shape[0]-self.steps
    
    
class ChannelMask(Dataset):
    def __init__(self, index=[0], seed=1, num=10, mean=None, std=None, scale=1, steps=1) -> None:
        super().__init__()
        set_seed(seed) # 设置采样种子
        self.steps = steps
        
        with open(f'{data_path}channelFlow/ch_2Dxysec.pickle', 'rb') as df:
            data = pickle.load(df)
            data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 2, 1) # (48, 128)
            self.data = data[index]  # n, c, h, w

        mask = (self.data < -0.5)|(self.data > 0.5)
        self.mean = mean if mean is not None else self.data[mask].mean()
        self.std = std if std is not None else self.data[mask].std() * scale
        self.data = (self.data - self.mean) / self.std
        
        # 使用高级索引创建稀疏数据张量
        self.size = self.data.shape[2:]
        positions = hw_sample(*self.size, num)
        self.observe = torch.zeros_like(self.data)
        self.observe[:, :, positions[:, 0], positions[:, 1]] = self.data[:, :, positions[:, 0], positions[:, 1]]

    def __getitem__(self, index):
        if self.steps < 2:
            return self.observe[index, :], self.data[index, :]
        else:
            end_index = index + self.steps
            features = self.observe[index:end_index, :]

            return features, self.data[end_index-1, :]

    def __len__(self):
        return self.data.shape[0]-self.steps
    

class ChannelVoronoi():
    def __init__(self, index=[0], seed=1, num=10, mean=None, std=None, scale=1, steps=1) -> None:
        np.random.seed(seed) # 设置采样种子
        self.steps = steps
        
        with open(f'{data_path}channelFlow/ch_2Dxysec.pickle', 'rb') as df:
            data = pickle.load(df)
            data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 2, 1) # (48, 128)
            self.data = data[index]  # n, c, h, w

        mask = (self.data < -0.5)|(self.data > 0.5)
        self.mean = mean if mean is not None else self.data[mask].mean()
        self.std = std if std is not None else self.data[mask].std() * scale
        self.data = (self.data - self.mean) / self.std
        
        h, w = self.data.shape[2:]
        self.size = (h, w)
        positions = hw_sample(h, w, num)
        
        x_coor, y_coor = np.linspace(0, w-1, w), np.linspace(0, h-1, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)
        
        # 使用高级索引提取数据
        sparse_data = self.data[:, 0, positions[:, 0], positions[:, 1]]
        
        # 最近邻插值，用方格坐标即可。实际位置中x-y方向等差均为0.08，与方格相同
        voronoilist = []
        for i in tqdm(range(len(index))):# 遍历快照索引
            voronoi = griddata(positions, sparse_data[i], (y_coor, x_coor), method='nearest')
            voronoilist.append(torch.from_numpy(voronoi).float())

        self.voronoi = torch.stack(voronoilist).unsqueeze(1) # 增加channel维度
        
        self.posmask = torch.zeros_like(self.voronoi)
        self.posmask[:, :, positions[:, 0], positions[:, 1]] = 1

    def __getitem__(self, index):
        # index为整数，返回无批次维度数据
        if self.steps < 2:
            return torch.cat([self.voronoi[index, :], self.posmask[index, :]], dim=0), self.data[index, :]
        else:
            end_index = index + self.steps
            features = torch.cat([self.voronoi[index:end_index, :], self.posmask[index:end_index, :]], dim=1)

            return features, self.data[end_index-1, :]

    def __len__(self):
        return self.data.shape[0]-self.steps