#!/usr/bin/python
# this file is expected to be executed in $CAFFE_ROOT
# no argument needed

from joblib import Parallel, delayed
import multiprocessing
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

os.system("cd $CAFFE_ROOT")
caffe_root = os.environ["CAFFE_ROOT"]
print caffe_root
sys.path.insert(0, caffe_root + '/python')
import caffe

def analyze_param(net, layers):
#   plt.figure()
    print '\n=============analyze_param start==============='
    total_nonzero = 0
    total_allparam = 0
    for i, layer in enumerate(layers):
        i += 1
        W = net.params[layer][0].data
        b = net.params[layer][1].data
#       plt.subplot(3, 1, i);
#       numBins = 2 ^ 8
#       plt.hist(W.flatten(), numBins, color='blue', alpha=0.8)
#       plt.show()
        print 'W(%d) range = [%f, %f]' % (i, min(W.flatten()), max(W.flatten()))
        print 'W(%d) mean = %f, std = %f' % (i, np.mean(W.flatten()), np.std(W.flatten()))
        non_zero = (np.count_nonzero(W.flatten()) + np.count_nonzero(b.flatten()))
        all_param = (np.prod(W.shape) + np.prod(b.shape))
        total_nonzero += non_zero
        total_allparam += all_param
        print 'non-zero W and b cnt = %d' % non_zero
        print 'total W and b cnt = %d' % all_param
        print 'percentage = %f\n' % (non_zero / float(all_param))
    print '=====> summary:'
    print 'non-zero W and b cnt = %d' % total_nonzero
    print 'total W and b cnt = %d' % total_allparam
    print 'percentage = %f' % (total_nonzero / float(total_allparam))
    print '=============analyze_param ends ==============='
    return total_nonzero / float(total_allparam)

# options defined here:
analyze_only = 0
target = 'lenet_300_100'
target = 'lenet5'
target = 'cifar'
target = 'imagenet'

if target == 'lenet_300_100':
    prototxt = caffe_root+'/mnist/pruning_lenet/lenet_300_100.prototxt'
    caffemodel = caffe_root+'/mnist/pruning_lenet/lenet_best_weights.caffemodel'
    net = caffe.Net(prototxt, caffemodel)
    layers = ['ip1', 'ip2', 'ip3']
    layers_tbd = ['ip1', 'ip2', 'ip3']
elif target == 'lenet5':
    prototxt = caffe_root+'/mnist/lenet.prototxt'
    caffemodel = caffe_root+'/mnist/lenet_iter_10000.caffemodel'
    net = caffe.Net(prototxt, caffemodel)
    layers = ['conv1', 'conv2', 'ip1', 'ip2']
    layers_tbd = ['ip1']
    output_prefix = caffe_root+'/mnist/lenet5_pruned'
elif target == 'cifar':
    prototxt = caffe_root+'/examples/cifar10/cifar10_full.prototxt'
    caffemodel = caffe_root+'/examples/cifar10/cifar10_full_iter_70000.caffemodel'
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    layers = ['conv1', 'conv2', 'conv3', 'ip1']
    layers_tbd = ['conv2', 'conv3', 'ip1']
    output_prefix = caffe_root+'/examples/cifar10/cifar_full_pruned'
elif target == 'imagenet':
    prototxt = caffe_root+'/3_prototxt_solver/train_val.prototxt'
    caffemodel = caffe_root+'/4_model_checkpoint/bvlc_alexnet.caffemodel'
#   net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    layers_tbd = [ 'fc6', 'fc7', 'fc8']
    output_prefix = caffe_root+'/4_model_checkpoint/1_before_retrain/alex_pruned_'



if analyze_only:
    analyze_param(net, layers)
    sys.exit(0)

num_cores = multiprocessing.cpu_count()
print "num_cores = %d" % num_cores
numBins = 2 ^ 8
threshold_list = [0.2e-6 * i for i in range(40)]
fout = open(caffe_root+'/2_results/parameter_cnt_678half.csv', 'w')
suffix = '_678half.caffemodel'

def calculate_gradients(net, n_iter=50):
    global layers_tbd
    grads = { l: None for l in layers_tbd }
    for i in range(n_iter):
        print 'Doing batch %d/%d' % (i, n_iter)
        net.forward()
        layer_diffs = net.backward()
        for layer in layers_tbd:
            if grads[layer] is None:
                grads[layer] = [net.params[layer][0].diff[...].copy(),
                                net.params[layer][1].diff[...].copy()]
            else:
                grads[layer][0] += net.params[layer][0].diff[...]
                grads[layer][1] += net.params[layer][1].diff[...]

    for layer in layers_tbd:
        grads[layer][0] /= n_iter
        grads[layer][1] /= n_iter

    return grads

def prune(threshold):
    global prototxt, caffemodel, output_prefix, layers, layers_tbd
    net = caffe.Net(prototxt, caffemodel, caffe.TRAIN)
    print '\n============  Calculating gradients  ============'
    grads = calculate_gradients(net)
    print '\n============  Surgery: threshold=%0.2f   ============' % threshold
    for i, layer in enumerate(layers_tbd):
        W = net.params[layer][0].data
        b = net.params[layer][1].data
        gradW = grads[layer][0]
        #     hi = np.max(np.abs(W.flatten()))
        hi = np.std(W.flatten())
#        print layer
#        histoBins = [0.2e-6*i for i in range(20)] + [1]
#        counts, edges = np.histogram(np.abs(W * gradW), bins=histoBins)
#        for c,e in zip(counts,edges):
#            print c, e
        mask = (np.abs(W * gradW) > (hi * threshold))
        if layer == 'fc8':
            mask = (np.abs(W) > (hi * threshold / 2))
        mask = np.bool_(mask)
        W = W * mask
        print 'non-zero W percentage = %0.4f ' % (np.count_nonzero(W.flatten()) / float(np.prod(W.shape)))
        net.params[layer][0].data[...] = W
        net.params[layer][0].mask[...] = mask
        print net.params[layer][0].mask.shape

    total_percentage = analyze_param(net, layers)
    output_model = output_prefix + str(threshold) + suffix
    net.save(output_model)
    return (threshold, total_percentage)

# Parallel version for this code:
# for threshold in threshold_list:
#     prune(threshold, layers, layers_tbd, fout, output_prefix)

caffe.set_mode_gpu()
caffe.set_device(1)
prune(0.0)

results = []
for threshold in threshold_list:
     results.append(prune(threshold))

#results = Parallel(n_jobs=num_cores)(delayed(prune)(threshold) for threshold in threshold_list)

print results
for (threshold, percentage) in results:
    fout.write("%4.1f, %.4f\n" % (threshold, percentage))
fout.close()
sys.exit(0)

