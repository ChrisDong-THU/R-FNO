import torch
from torch.utils.data import DataLoader

import argparse
import logging

from model import FNO2d
from data import dataset_dict
from utils import loss_dict, prep_experiment, save_model, plot_result, update_ckpt

parser = argparse.ArgumentParser(description='Field reconstruction exp.')
# 训练参数
parser.add_argument('--exp', type=str, default="test")
parser.add_argument('--cfg', type=str, default="cnn-1")
parser.add_argument('--snapshot', action='store_true', help="save the best snapshot or not")
args = parser.parse_args()

def train(args):
    tb_writer, cfg = prep_experiment(args)
    torch.cuda.set_device(cfg.train.gpu)
    torch.backends.cudnn.benchmark = True
    ckpt = {'epoch': -1, 'loss': 1e10, 'path': cfg.logs.ckpt_path}
    
    # 数据加载器
    train_index = [i for i in range(4000)]
    val_index = [i for i in range(4000, 5000)]
    Dataset = dataset_dict.get(cfg.data.type).get(cfg.data.embed)
    
    train_dataset = Dataset(index=train_index, seed=cfg.data.seed, num=cfg.data.num)
    mean, std = train_dataset.mean, train_dataset.std
    val_dataset = Dataset(index=val_index, seed=cfg.data.seed, num=cfg.data.num, mean=mean, std=std)
    in_channels = 1 if cfg.data.embed=='mask' else 2
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, num_workers=4) # 不打乱，用最后同一张图像对比

    # 实例化模型
    net = FNO2d(in_channels=in_channels, modes1=cfg.fno.modes1, modes2=cfg.fno.modes2, width=cfg.fno.width).cuda()
    net.train()
    no_improve_val_epoch = 0 # 早停计数

    loss_function = loss_dict.get(cfg.train.loss)
    
    # 设置优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.train.lr)
    if cfg.train.scheduler == 'e':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.train.e_scheduler_gamma)
    elif cfg.train.scheduler == 'ms':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.train.ms_milestone, gamma=cfg.train.ms_scheduler_gamma)
    else:
        raise ValueError("Unsupported scheduler")
    
    for epoch in range(cfg.train.epochs):
        train_loss, train_num = 0., 0.
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            pre = net(inputs)
            loss = loss_function(labels, pre)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.shape[0]
            train_num += inputs.shape[0]

        train_loss = train_loss / train_num
        # Tensorboard记录
        current_lr = scheduler.get_last_lr()[0]
        tb_writer.add_scalar('Learning rate/lr', current_lr, epoch)
        tb_writer.add_scalar('Loss/train', train_loss, epoch)
        logging.info("Epoch: {}, train_loss: {}".format(epoch, train_loss))
        if cfg.train.scheduler == 'ms' or epoch < cfg.train.e_milestone:
            scheduler.step()  # 调整学习率

        if epoch % cfg.train.val_interval == 0:
            net.eval()
            val_loss, val_num = 0., 0.
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.cuda(), labels.cuda()
                with torch.no_grad():
                    pre = net(inputs)
                loss = loss_function(labels, pre)

                val_loss += loss.item() * inputs.shape[0]
                val_num += inputs.shape[0]

            # Tensorboard记录
            val_loss = val_loss / val_num
            tb_writer.add_scalar('Loss/val', val_loss, epoch)
            logging.info("Epoch: {}, val_loss: {}".format(epoch, val_loss))
            if val_loss < ckpt['loss']:
                ckpt = save_model(ckpt, epoch, val_loss, net, optimizer) if args.snapshot else update_ckpt(ckpt, epoch, val_loss)
                no_improve_val_epoch = 0
            else:
                no_improve_val_epoch += 1

            if no_improve_val_epoch >= cfg.train.val_patience:
                logging.info("Early stopping triggered at epoch {}".format(epoch))
                break

            net.train()

            # 绘制最后的验证集恢复情况
            if epoch % cfg.monitor.plot_freq == 0:
                plot_result(labels[-1, 0, :, :].cpu().numpy(), pre[-1, 0, :, :].cpu().numpy(),
                        file_name=cfg.logs.fig_path + f'/epoch{epoch}.png')


if __name__ == '__main__':
    try:
        train(args)
    except KeyboardInterrupt:
        print('Training stopped by user.')