import torch
from torch.utils.data import DataLoader

import argparse
from omegaconf import OmegaConf
from tqdm import tqdm

from model import FNO2d, RFNO, RecFieldCNN
from data import dataset_dict
from data.config import sensorset
from utils import plot_result, load_model, eval_MRE

parser = argparse.ArgumentParser(description='field reconstruction')
parser.add_argument('--ckpt', type=str, default='ckpt',
                    help='path to the check_point')
parser.add_argument('--index', type=str, default='1000-2000',
                    help='range of indices, formatted as start-end')
parser.add_argument('--cfg', type=str, default='fno-cy-1')
args = parser.parse_args()


def eval(args):
    cfg = OmegaConf.load(f'./conf/{args.cfg}.yaml')
    # 准备数据
    start, end = map(int, args.index.split('-'))
    test_index = [i for i in range(start, end)]
    Dataset = dataset_dict.get(cfg.data.type).get(cfg.data.embed)
    dataset = Dataset(index=test_index, seed=cfg.data.seed, num=cfg.data.num, steps=cfg.data.get('steps', 1))
    img_size = dataset.size
    in_channels = 1 if cfg.data.embed == 'mask' else 2
    # 准备数据
    eval_loader = DataLoader(dataset, batch_size=16, num_workers=4)

    # 加载模型
    # net = FNO2d(in_channels=in_channels, modes1=cfg.fno.modes1, modes2=cfg.fno.modes2, width=cfg.fno.width).cuda()
    # net = RFNO(in_channels=in_channels, img_size=img_size, fno=cfg.fno, gru=cfg.gru, sync=cfg.rfno.sync,
    #            model_path="test.pth").cuda()
    net = RecFieldCNN(in_channels=in_channels).cuda()

    ckpt = load_model(args.ckpt)
    net.load_state_dict(ckpt['model'])
    print(f'>>> load model {args.ckpt} ...')
    net.eval()

    # 均方误差，测试样本数
    eval_mre, eval_num = 0.0, 0.0
    for i, (inputs, labels) in tqdm(enumerate(eval_loader), total=len(eval_loader)):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            pre = net(inputs)
        eval_mre += eval_MRE(gt=labels[:, 0, :], pre=pre[:, 0, :]) * inputs.shape[0]
        eval_num += inputs.shape[0]

    eval_mre = eval_mre / eval_num
    print(f'>>> eval num: {eval_num}, eval MRE: {eval_mre}')

    plot_result(labels[-1, 0, :].cpu().numpy(), pre[-1, 0, :].cpu().numpy(),
                f'./eval_{args.index}.png')


if __name__ == '__main__':
    eval(args)
