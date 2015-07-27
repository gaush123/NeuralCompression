'''
Created on Apr 18, 2015

@author: huizi
'''

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.vq as scv
import pickle
from kmeans import *

caffe.set_device(3)
log = log + '.layer'
def eval_accu_layerwise(prototxt, caffemodel, bits_list, log):
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())
    layers = map(lambda x: [x], layers)
    layers.append(filter(lambda x: 'conv' in x, net.params.keys()))
    layers.append(filter(lambda x: 'fc' in x or 'ip' in x, net.params.keys()))
    
    accu_top1 = np.zeros((len(layers), len(bits_list)))
    accu_top5 = np.zeros((len(layers), len(bits_list)))
    for i, layer in enumerate(layers):
        for j, bits in enumerate(bits_list):
            num_c = 2 ** bits

            net = caffe.Net(prototxt, caffemodel, caffe.TEST)
            print "==================Notes================"
            print "Current proceedings:", layer, "bits:", bits

            codebook = kmeans_net(net, layer, num_c)

            quantize_net(net, codebook)

            net.save(caffemodel + '.layer.quantize')
            log_new = log + "_layer%d"%i + "_%dbits"%bits
            command = caffe_root + "/build/tools/caffe test --model=" + prototxt + " --weights=" + caffemodel + ".layer.quantize --iterations=%d --gpu 2 2>"%iters +log_new
            print command
            os.system(command)
            os.system('tail -n 3 '+ log_new)

            accu_top1[i,j], accu_top5[i,j] = parse_caffe_log(log_new)
    return (accu_top1, accu_top5)

def main1():
    bits_list = [2,3,4,8]

    accu_top1, accu_top5 = eval_accu_layerwise(prototxt, caffemodel, bits_list, log)
    np.save(dir_t + 'accu_top1', accu_top1)
    np.save(dir_t + 'accu_top5', accu_top5)

if __name__ == "__main__":
    main1()
