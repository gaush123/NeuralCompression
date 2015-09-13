'''
Created on Apr 18, 2015

@author: huizi
'''

import sys
import os
import numpy as np
import matplotlib.pyplot as plt


# os.system("cd $CAFFE_ROOT")
caffe_root = os.environ["CAFFE_ROOT"]
os.chdir(caffe_root)
print caffe_root
sys.path.insert(0, caffe_root + 'python')
import caffe

banks = 64
buffer_number = 4
jump_bits = 4
data_bits = 4
ptr_bits = 32
line_bits = 128

option = 'alexnet'
if option == 'alexnet':
    prototxt = '3_prototxt_solver/L2/train_val.prototxt'
    solver_proto = '3_prototxt_solver/L2/finetune_solver.prototxt'
    caffemodel = '4_model_checkpoint/alexnet/alexnet9x.caffemodel'
elif option == 'vgg':
    prototxt = '3_prototxt_solver/vgg16/train_val.prototxt'
    caffemodel = '4_model_checkpoint/vgg16/vgg16_12x.caffemodel'
    solver_proto = '3_prototxt_solver/L2/finetune_solver.prototxt'
    snap_dir = '4_model_checkpoint/4_model_checkpoint/vgg16/snapshot/'

net = caffe.Net(prototxt, caffemodel, caffe.TEST)


layers = filter(lambda x: 'fc' in x, net.params.keys())
nonzeros_layers = map(lambda x:np.count_nonzero(net.params[x][0].data),layers) 

buffer_size_per_bank = buffer_number * 32 / 8

total_nonzeros = sum(nonzeros_layers)
data_size_per_bank = data_bits * total_nonzeros / 8 / banks
loc_size_per_bank = jump_bits * total_nonzeros / 8 / banks

ptr_size_per_bank = sum(map(lambda x:net.params[x][0].data.shape[0], layers)) * 2 * ptr_bits / 8

act_size_per_bank = max(map(lambda x:net.params[x][0].data.shape[1], layers)) * 2 * 32 / banks / 8

print "average data/ptr capacity:", float(total_nonzeros/banks) / 2**ptr_bits
print "data size per bank(Bytes):", data_size_per_bank
print "loc size per bank(Bytes):", loc_size_per_bank
print "ptr size per bank(Bytes):", ptr_size_per_bank
print "act size per bank(Bytes):", act_size_per_bank
print 
total_size = banks * (data_size_per_bank + loc_size_per_bank + ptr_size_per_bank * 2 + act_size_per_bank)
print "total size:(Bytes):", total_size
