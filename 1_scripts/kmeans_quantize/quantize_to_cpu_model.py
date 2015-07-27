'''
Created on Apr 18, 2015

@author: huizi, songhan
'''

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.vq as scv
import pickle

# os.system("cd $CAFFE_ROOT")
caffe_root = os.environ["CAFFE_ROOT"]
os.chdir(caffe_root)
print caffe_root
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()
caffe.set_device(1)
option = 'alexnet'
if option == 'lenet5':
    prototxt = '3_prototxt_solver/lenet5/train_val.prototxt'
    caffemodel = '4_model_checkpoint/lenet5/lenet5.caffemodel'
    iters = 100
    dir_t = '2_results/kmeans/lenet5/'
elif option == 'alexnet':
    prototxt = '3_prototxt_solver/L2/train_val.prototxt'
    caffemodel = '4_model_checkpoint/alexnet/alexnet9x.caffemodel'
    iters = 1000
    dir_t = '2_results/kmeans/alexnet/'
elif option == 'vgg':
    prototxt = '3_prototxt_solver/vgg16/train_val.prototxt'
    caffemodel = '4_model_checkpoint/vgg16/vgg16_12x.caffemodel'
    iters = 1000
    dir_t = '2_results/kmeans/vgg16/'

log = dir_t + 'log_accu'

def kmeans_net(net, layers, num_c=16, initials=None):
    codebook = {}
    if type(num_c) == type(1):
        num_c = [num_c] * len(layers)
    else:
        assert len(num_c) == len(layers)

    print "==============Perform K-means============="
    for idx, layer in enumerate(layers):
        print "Eval layer:", layer
        W = net.params[layer][0].data.flatten()
        W = W[np.where(W != 0)]
        if initials is None:  # Default: uniform sample
            std = np.std(W)
            initial_uni = np.linspace(-4 * std, 4 * std, num_c[idx] - 1)
            codebook[layer], _ = scv.kmeans(W, initial_uni)
            '''
            codebook[layer],_= scv.kmeans(W, num_c[idx] - 1)
            '''
        elif type(initials) == type(np.array([])):
            codebook[layer], _ = scv.kmeans(W, initials)
        else:
            print type(initials)
            return None
        codebook[layer] = np.append(0.0, codebook[layer])
        print "codebook size:", len(codebook[layer])
    return codebook

def quantize_update_net(net, codebook):
    layers = codebook.keys()
    print "================Perform quantization=============="
    for layer in layers:
        print "Quantize layer:", layer
        W = net.params[layer][0].data
        codes, dist = scv.vq(W.flatten(), codebook[layer])
        W_q = np.reshape(codebook[layer][codes], W.shape)
        mask_codebook = np.reshape(codes, W.shape)
        np.copyto(net.params[layer][0].data, W_q)
        np.copyto(net.params[layer][0].mask_codebook, mask_codebook)
        net.params[layer][0].codebook[:len(codebook[layer])] = codebook[layer]

def parse_caffe_log(log):
    lines = open(log).readlines()
    return map(lambda x: float(x.split()[-1]), lines[-3:-1])


def main(choice=[64, 16]):
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
# layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())
    layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())

    num_c = [choice[0]] * (len(layers) - 3) + [choice[1]] * 3
    codebook = kmeans_net(net, layers, num_c)


    quantize_update_net(net, codebook)

    net.save(caffemodel + '.update')

if __name__ == "__main__":
    main([256, 16])
