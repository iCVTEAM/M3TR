from thop import *
from models import get_model
import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch Training for Multi-label Image Classification')

''' Fixed in general '''
parser.add_argument('--data_root_dir', default='./datasets/', type=str, help='save path')
parser.add_argument('--image-size', '-i', default=448, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+', help='number of epochs to change learning rate')
# parser.add_argument('--device_ids', default=[0], type=int, nargs='+', help='number of epochs to change learning rate')
parser.add_argument('-b', '--batch-size', default=32, type=int)
parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='INT', help='number of data loading workers (default: 4)')
parser.add_argument('--display_interval', default=200, type=int, metavar='M', help='display_interval')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float)
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float, metavar='LRP', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--max_clip_grad_norm', default=10.0, type=float, metavar='M', help='max_clip_grad_norm')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')

''' Train setting '''
parser.add_argument('--data', metavar='NAME', help='dataset name (e.g. COCO2014')
parser.add_argument('--model_name', type=str, default='MLVIT')
parser.add_argument('--save_dir', default='./checkpoint/COCO2014/', type=str, help='save path')

''' Val or Tese setting '''
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')


model = get_model(80, args=parser.parse_args())
input = torch.randn(1,3,448, 448)
flops, params = profile(model, inputs=(input, ))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')
