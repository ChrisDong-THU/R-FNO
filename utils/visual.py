import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_mask(field1, field2, file_name='test.png'):
    size = field1.shape
    x, y = np.linspace(0, size[1], size[1]), np.linspace(0, size[0], size[0])
    x, y = np.meshgrid(x, y)

    plt.figure(figsize=(10*size[1]/size[0], 5))
    plt.subplot(1, 2, 1)
    plt.contourf(x, y, field1, levels=2, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(orientation='horizontal')
    plt.subplot(1, 2, 2)
    plt.contourf(x, y, field2, levels=100, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(orientation='horizontal')
    
    # 减少子图之间的空白区域大小
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_voronoi(field1, field2, field3, file_name='test.png'):
    size = field1.shape
    x, y = np.linspace(0, size[1], size[1]), np.linspace(0, size[0], size[0])
    x, y = np.meshgrid(x, y)

    plt.figure(figsize=(13*size[1]/size[0], 5))
    plt.subplot(1, 3, 1)
    plt.contourf(x, y, field1, levels=100, cmap='binary', vmin=0, vmax=1)
    plt.colorbar(orientation='horizontal')
    plt.subplot(1, 3, 2)
    plt.contourf(x, y, field2, levels=100, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(orientation='horizontal')
    plt.subplot(1, 3, 3)
    plt.contourf(x, y, field3, levels=100, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(orientation='horizontal')
    
    # 减少子图之间的空白区域大小
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_result(field1, field2, file_name='test.png'):
    size = field1.shape
    x, y = np.linspace(0, size[1], size[1]), np.linspace(0, size[0], size[0])
    x, y = np.meshgrid(x, y)

    plt.figure(figsize=(13*size[1]/size[0], 5))
    plt.subplot(1, 3, 1)
    plt.contourf(x, y, field1, levels=100, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(orientation='horizontal')
    plt.subplot(1, 3, 2)
    plt.contourf(x, y, field2, levels=100, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(orientation='horizontal')
    plt.subplot(1, 3, 3)
    plt.contourf(x, y, field1-field2, levels=100, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(orientation='horizontal')
    
    # 减少子图之间的空白区域大小
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
def video1x3(field1, field2, field3, video_name='test.mp4'):
    batch, h, w = field1.shape
    fig, axs = plt.subplots(1, 3, figsize=(15*w/h, 5))
    # 自动调整子图参数
    plt.tight_layout(pad=0.3, h_pad=0.3, w_pad=0.3)

    pbar = tqdm(total=batch)
    def update(i):
        axs[0].contourf(field1[i], levels=100, cmap='binary', vmin=0, vmax=1)
        axs[1].contourf(field2[i], levels=100, cmap='RdBu_r', vmin=-1, vmax=1)
        axs[2].contourf(field3[i], levels=100, cmap='RdBu_r', vmin=-1, vmax=1)
        # 更新进度条
        pbar.update(1)
        return axs
    
    anim = animation.FuncAnimation(fig, update, frames=batch, blit=False)
    
    # 保存为视频
    anim.save(video_name, writer='ffmpeg', fps=1)

    plt.close(fig)