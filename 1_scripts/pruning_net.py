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
sys.path.insert(0, caffe_root + 'python')
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
folder="/L1_2/"

prototxt = caffe_root+'/3_prototxt_solver/'+folder+'deploy.prototxt'
caffemodel = caffe_root+'/4_model_checkpoint/0_original_dense/'+folder+'bvlc_alexnet.caffemodel'
if folder[2]=='1':
    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6_new', 'fc7_new', 'fc8_new']
    # layers_tbd = [ 'fc6_new', 'fc7_new', 'fc8_new']
    layers_tbd = ['fc6']
if folder[2]=='2':
    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    # layers_tbd = [ 'fc6', 'fc7', 'fc8']
    layers_tbd = ['fc6']
else:
    print "error"
suffix = '_fc6.caffemodel'
output_prefix = caffe_root+'/4_model_checkpoint/1_before_retrain/'+folder+'layerwise_'
threshold_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3]
fout = open(caffe_root+'/2_results/'+folder+'parameter_cnt_678half.csv', 'w')


numBins = 2 ^ 8
if analyze_only:
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    analyze_param(net, layers)
    sys.exit(0)


num_cores = multiprocessing.cpu_count()
print "num_cores = %d" % num_cores
def prune(threshold):
    global prototxt, caffemodel, output_prefix, layers, layers_tbd
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n============  Surgery: threshold=%0.2f   ============' % threshold
    for i, layer in enumerate(layers_tbd):
        W = net.params[layer][0].data
        b = net.params[layer][1].data
        #     hi = np.max(np.abs(W.flatten()))
        hi = np.std(W.flatten())
        mask = (np.abs(W) > (hi * threshold))
        # if layer == layers_tbd[-1]:
        #     mask = (np.abs(W) > (hi * threshold / 2))
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

results = Parallel(n_jobs=num_cores)(delayed(prune)(threshold) for threshold in threshold_list)

print results
for (threshold, percentage) in results:
    fout.write("%4.1f, %.4f, \n" % (threshold, percentage))
fout.close()
sys.exit(0)

