import torch
from torch.utils.data import DataLoader

import sys
import os

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(filename)

from data import dataset_dict
from data.config import sensorset, ss_sample
from utils import plot_voronoi, plot_mask, video1x3

index = [i for i in range(100)]
Dataset = dataset_dict.get('ch').get('voronoi')
dataset = Dataset(index=index, seed=400, num=200)
loader = DataLoader(dataset, batch_size=30, shuffle=False)
data = dataset[0]
gt = data[1][0]

def test_mask():
    mask = data[0][0]
    plot_mask(mask, gt)

def test_voronoi():
    vn = data[0][0]
    pm = data[0][1]
    plot_voronoi(pm, vn, gt)

# test_mask()
# test_voronoi()
pass
# print(data[0].shape, data[1].shape) # 特征项形状，标签项形状

for i, (inputs, labels) in enumerate(loader):
    field1 = inputs[:,1,:]
    field2 = inputs[:,0,:]
    field3 = labels[:,0,:]
    video1x3(field1=field1, field2=field2, field3=field3, video_name=f'test{i}.mp4')
    break
