import os
from datetime import datetime, timedelta, timezone
import logging
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

loss_dict = {
    'l1': F.l1_loss,
    'mse': F.mse_loss
}

def save_log(prefix, output_dir, time_str):
    fmt = '%(asctime)s.%(msecs)03d %(message)s'
    date_fmt = '%m-%d %H:%M:%S'
    filename = os.path.join(output_dir, prefix + '_' + time_str + '.log')
    print("Logging :", filename)
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=date_fmt,
                        filename=filename, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    console.setFormatter(formatter)
    # 控制台输出且文件记录
    logging.getLogger('').addHandler(console)

def prep_experiment(args):
    """
    Make output directories, setup logging, Tensorboard, snapshot code.
    """
    cfg = OmegaConf.load(f'./conf/{args.cfg}.yaml')
    exp = f'logs/{cfg.data.type}/{cfg.data.embed}/{args.exp}'
    cfg.logs.log_path = os.path.join(exp, cfg.logs.log_path)
    cfg.logs.sw_path = os.path.join(exp, cfg.logs.sw_path)
    cfg.logs.fig_path = os.path.join(exp, cfg.logs.fig_path)
    cfg.logs.ckpt_path = os.path.join(exp, cfg.logs.ckpt_path)
    
    os.makedirs(cfg.logs.log_path, exist_ok=True)
    os.makedirs(cfg.logs.sw_path, exist_ok=True)
    os.makedirs(cfg.logs.fig_path, exist_ok=True)
    os.makedirs(cfg.logs.ckpt_path, exist_ok=True)
    
    timestr = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime('%Y_%m_%d_%H_%M_%S')
    save_log('log', cfg.logs.log_path, timestr)
    open(os.path.join(cfg.logs.log_path, timestr + '.txt'), 'w').write(str(args) + '\n')
    writer = SummaryWriter(log_dir=cfg.logs.sw_path, comment=timestr)
    
    return writer, cfg

def update_ckpt(ckpt, epoch, loss):
    # save new best
    ckpt['epoch'] = epoch
    ckpt['loss'] = loss
    logging.info("Epoch: {}, improved".format(epoch))
    
    return ckpt

def save_model(ckpt, epoch, loss, model, optimizer):
    '''保存模型

    :param dict ckpt: 检查点ckpt
    :param int epoch: 轮次
    :param float loss: 损失
    :param Moudle model: 模型
    :return dict: 检查点
    '''
    # remove old models
    if epoch > 0:
        best_snapshot = 'ckpt_epoch_{}_loss_{:.8f}'.format(
            ckpt['epoch'], ckpt['loss'])
        best_snapshot = os.path.join(ckpt['path'], best_snapshot)
        assert os.path.exists(best_snapshot), 'cant find old snapshot {}'.format(best_snapshot)
        os.remove(best_snapshot)

    ckpt = update_ckpt(ckpt, epoch, loss)

    best_snapshot = 'ckpt_epoch_{}_loss_{:.8f}'.format(
        ckpt['epoch'], ckpt['loss'])
    best_snapshot = os.path.join(ckpt['path'], best_snapshot)

    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint,best_snapshot)
    
    return ckpt

def load_model(ckpt_path):
    assert os.path.isfile(ckpt_path), f'cant find checkpoint {ckpt_path}'
    checkpoint = torch.load(ckpt_path)
    logging.info("loaded checkpoint at epoch {}".format(checkpoint['epoch']))
    
    return checkpoint

def eval_MRE(gt, pre):
    '''计算批量二维场相对误差
    
    :param tensor gt: 形状(batch_size, h, w)
    '''
    # 计算每个样本的预测误差的 L2 范数
    error_norm = torch.norm(gt - pre, dim=[1,2])
    
    # 计算每个样本的真值的 L2 范数
    gt_norm = torch.norm(gt, dim=[1,2])
    
    # 计算相对误差
    epsilon = error_norm / gt_norm
    
    # 计算整个批次的平均相对误差
    mean_epsilon = torch.mean(epsilon)
    
    return mean_epsilon