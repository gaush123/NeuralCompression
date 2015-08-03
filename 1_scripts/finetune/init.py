import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.vq_maohz as scv
import pickle
import argparse
import time

parser = argparse.ArgumentParser()
#===========Basic options=============
parser.add_argument('--device-id', '-d', dest='device_id', type=int, default=0)
parser.add_argument('bits', type=int, nargs='+')
parser.add_argument('--layers', dest='layers', type=str, default='all')
parser.add_argument('--network', dest='network', type=str, default='alexnet')
parser.add_argument('--timing', dest='timing', action='store_true')

parser.add_argument('--kmeans-method', dest='kmeans_method', type=str, default='linear')
parser.add_argument('--no-kmeans-compress', dest='kmeans_compress', action='store_false')

#=======Training options===============
parser.add_argument('--lr', dest='lr', type=float, default=50)
parser.add_argument('--decay-rate', dest='decay_rate', type=float, default=0.99)
parser.add_argument('--momentum', dest='momentum', type=float, default=0.0)
parser.add_argument('--update', dest='update', type=str, default='sgd')

parser.add_argument('--normalize-codebook-diff', dest='normalize_flag', action='store_true')
parser.add_argument('--no-average-diff', dest='average_flag', action='store_false')

#===========For iterative training==============
parser.add_argument('--finetune-codebook-iters', dest='co_iters', type=int, default=1)
parser.add_argument('--accumulate-diff-iters', dest='ac_iters', type=int, default=10)
parser.add_argument('--stochastic', dest='use_stochastic', action='store_true')

#=============For Snapshot=====================
parser.add_argument('--snapshot', dest='snapshot', type=str, default='None')

parser.set_defaults(timing=False, kmeans_compress=True, normalize_flag=False, average_flag=True, use_stochastic=False)

args = parser.parse_args()

#=================Initializa Caffe==============
caffe_root = os.environ["CAFFE_ROOT"]
os.chdir(caffe_root)
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(args.device_id)

#==============Set paths========================
option = args.network
if option == 'lenet5':
    prototxt = '3_prototxt_solver/lenet5/train_val.prototxt'
    caffemodel = '4_model_checkpoint/lenet5/lenet5.caffemodel'
    solver_proto = '3_prototxt_solver/lenet5/finetune_solver.prototxt'
    iters = 100
    dir_t = '2_results/kmeans/lenet5/'
    snap_dir = '4_model_checkpoint/lenet5/snapshot/'
elif option == 'alexnet':
    prototxt = '3_prototxt_solver/L2/train_val.prototxt'
    solver_proto = '3_prototxt_solver/L2/finetune_solver.prototxt'
    caffemodel = '4_model_checkpoint/alexnet/alexnet9x.caffemodel'
    iters = 1000
    dir_t = '2_results/kmeans/alexnet/'
    snap_dir = '4_model_checkpoint/alexnet/snapshot/'
elif option == 'vgg':
    prototxt = '3_prototxt_solver/vgg16/train_val.prototxt'
    caffemodel = '4_model_checkpoint/vgg16/vgg16_12x.caffemodel'
    solver_proto = '3_prototxt_solver/vgg16/finetune_solver.prototxt'
    iters = 1000
    dir_t = '2_results/kmeans/vgg16/'
elif option == 'lenet_300':
    prototxt = '3_prototxt_solver/lenet_300_100/train_val.prototxt'
    caffemodel = '4_model_checkpoint/lenet_300_100/lenet300_100_9x.caffemodel'
    solver_proto = '3_prototxt_solver/lenet_300_100/finetune_solver.prototxt'
    iters = 100
    dir_t = '2_results/kmeans/lenet_300_100/'
log = dir_t + 'log_accu'

#==================Initializa solver and net==========
solver = caffe.SGDSolver(solver_proto)

solver.net.copy_from(caffemodel)
net = solver.net

total_layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())
if args.layers == 'all':
    update_layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())
else:
    update_layers = filter(lambda x:args.layers in x, net.params.keys())

if len(args.bits) == 1:
    num_c = args.bits * len(total_layers)
elif len(args.bits) == 2:
    num_conv = len(filter(lambda x:'conv' in x, total_layers))
    num_c = [args.bits[0]] * (num_conv) + [args.bits[1]] * (len(total_layers) - num_conv)
else:
    num_c = args.bits
    assert len(num_c) == len(total_layers)
num_c = map(lambda x: 2 ** x, num_c)


