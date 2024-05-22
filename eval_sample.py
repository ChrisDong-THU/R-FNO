import torch

import argparse
from omegaconf import OmegaConf

from model import FNO2d, RFNO
from data import dataset_dict
from data.config import sensorset
from utils import plot_result, load_model, eval_MRE

parser = argparse.ArgumentParser(description='field reconstruction')
parser.add_argument('--ckpt', type=str, default='ckpt',
                    help='path to the check_point')
parser.add_argument('--index', type=int, default=1000)
parser.add_argument('--cfg', type=str, default='fno-1')
args = parser.parse_args()


def eval(args):
    cfg = OmegaConf.load(f'./conf/{args.cfg}.yaml')
    # 准备数据
    test_index = [i for i in range(5000)]
    Dataset = dataset_dict.get(cfg.data.type).get(cfg.data.embed)
    dataset = Dataset(index=test_index, seed=cfg.data.seed, num=cfg.data.num, steps=cfg.data.steps)
    inputs, labels = dataset[args.index]

    # 补上批次维度
    inputs = inputs.unsqueeze(0)
    labels = labels.unsqueeze(0)

    # 加载模型
    net = FNO2d().cuda()
    ckpt = load_model(args.ckpt)
    net.load_state_dict(ckpt['model'])
    print(f'>>> load model {args.model} ...')
    net.eval()

    inputs, labels = inputs.cuda(), labels.cuda()
    with torch.no_grad():
        pre = net(inputs)
    eval_mre = eval_MRE(gt=labels[:,0,:], pre=pre[:,0,:])

    print(f'>>> eval MRE: {eval_mre}')
    plot_result(labels[0, 0, :].cpu().numpy(), pre[0, 0, :].cpu().numpy(), f'./eval_{args.index}_{args.sensorset}.png')


if __name__ == '__main__':
    eval(args)
