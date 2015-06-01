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
print "caffe_root is: ", caffe_root
sys.path.insert(0, caffe_root + 'python')
import caffe

def analyze_param(net, layers):
#   plt.figure()
    print '\n=============analyze_param start==============='
    total_nonzero = 0
    total_allparam = 0
    percentage_list = []
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
        this_layer_percentage = non_zero / float(all_param)
        total_nonzero += non_zero
        total_allparam += all_param
        print 'non-zero W and b cnt = %d' % non_zero
        print 'total W and b cnt = %d' % all_param
        print 'percentage = %f\n' % (this_layer_percentage)
        percentage_list.append(this_layer_percentage)
    print '=====> summary:'
    print 'non-zero W and b cnt = %d' % total_nonzero
    print 'total W and b cnt = %d' % total_allparam
    print 'percentage = %f' % (total_nonzero / float(total_allparam))
    print '=============analyze_param ends ==============='
    return (total_nonzero / float(total_allparam), percentage_list)

# options defined here:
analyze_only = 0
folder = "/vgg16/"

prototxt = caffe_root + '/3_prototxt_solver/' + folder + 'train_val.prototxt'
caffemodel = caffe_root + '/4_model_checkpoint/0_original_dense' + folder + 'VGG_ILSVRC_16_layers.caffemodel'
layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
layers_tbd = [ 'fc6', 'fc7', 'fc8']
layers = layers_tbd

suffix = 'fc678_1.5'
suffix_2 = 'pruned_afterConv_'
output_prefix = caffe_root + '/4_model_checkpoint/1_before_retrain/' + folder + suffix_2
threshold_list = [0, 0.4, 2.0, 2.1, 2.42]
threshold_list = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
threshold_list = [1.22, 1.24, 1.26, 1.86, 1.88, 2.62, 2.64, 2.66, 2.68, 2.72, 2.74]
threshold_list = [2.81, 2.82, 2.83, 2.84, 2.85, 2.86, 2.87, 2.88, 2.89, 2.9]
threshold_list = [0.4, 1.21, 1.6, 1.88, 2.1, 2.42, 2.55, 2.66, 2.76, 2.85]
threshold_list = np.arange(0.1,2.1,0.1)

print "threshold list is", threshold_list
fout = open(caffe_root + '/2_results/' + folder + 'parameter_cnt_' + suffix + '.csv', 'a')
fout2 = open(caffe_root + '/2_results/' + folder + 'eachLayer_' + suffix + '.csv', 'a')

# threshold_list = [ "{:1.2f}".format(x) for x in threshold_list]
# print "threshold list is", threshold_list



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
        # hi = np.max(np.abs(W.flatten()))
        hi = np.std(W.flatten())
	local_threshold = threshold
        # local_threshold is different for each layer:
        if layer == layers_tbd[0]:
            local_threshold = threshold
        if layer == layers_tbd[1]:
            local_threshold = threshold
        if layer == layers_tbd[2]:
            local_threshold = threshold / 1.5
            if local_threshold > 1.3: local_threshold = 1.3

        mask = (np.abs(W) > (hi * local_threshold))
        mask = np.bool_(mask)
        W = W * mask
        print 'non-zero W percentage = %0.5f ' % (np.count_nonzero(W.flatten()) / float(np.prod(W.shape)))
        net.params[layer][0].data[...] = W
        net.params[layer][0].mask[...] = mask
        print net.params[layer][0].mask.shape

    (total_percentage, percentage_list) = analyze_param(net, layers)
    output_model = output_prefix + str(threshold) + '_' + suffix + ".caffemodel"
    net.save(output_model)
    return (threshold, total_percentage, percentage_list)

# Parallel version for this code:
# for threshold in threshold_list:
#     prune(threshold, layers, layers_tbd, fout, output_prefix)

results = Parallel(n_jobs=num_cores)(delayed(prune)(threshold) for threshold in threshold_list)

for result in results:
    print result
for (threshold, total_percentage, percentage_list) in results:
    fout.write("%4.2f, %.5f,\n" % (threshold, total_percentage))

for (threshold, total_percentage, percentage_list) in results:
    fout2.write("%4.2f, %.5f, %.5f, %.5f\n" % (threshold, percentage_list[-3], percentage_list[-2], percentage_list[-1]))

fout.close()
fout2.close()
sys.exit(0)

