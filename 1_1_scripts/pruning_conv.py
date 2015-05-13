#!/usr/bin/python
# no argument needed

from joblib import Parallel, delayed
import multiprocessing
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('../')
sys.path.insert(0,'./python/')
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
save_net = 1
folder = "test/"

prototxt = "VGG_11/VGG_ILSVRC_16_layers_deploy.prototxt"
caffemodel = "VGG_11/VGG_ILSVRC_16_layers.caffemodel"
# layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
# layers_tbd = [ 'conv1', 'conv2', 'conv3', 'conv4', 'conv5']

suffix = 'all'
suffix_2 = 'conv_all_'
output_prefix = folder + suffix_2
# threshold_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3]
threshold_list = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# threshold_list = np.arange(2.00, 3.00, 0.01)

print "threshold list is", threshold_list
fout = open('cva_scripts/' + folder + 'parameter_cnt_' + suffix + '.csv', 'w')
fout2 = open('cva_scripts/' + folder + 'eachLayer_' + suffix + '.csv', 'w')

# threshold_list = [ "{:1.2f}".format(x) for x in threshold_list]
# print "threshold list is", threshold_list

def get_layers(net):
    layers = [layer for layer in net.params.keys() if 'conv' in layer or 'fc' in layer]
    layers_tbd = [layer for layer in layers if 'conv' in layer]
    return (layers, layers_tbd)

numBins = 2 ^ 8
if analyze_only:
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    layers, layers_tbd = get_layers(net)
    analyze_param(net, layers)
    sys.exit(0)


num_cores = multiprocessing.cpu_count() -5 
print "num_cores = %d" % num_cores
def prune(threshold):
    global prototxt, caffemodel, output_prefix, layers, layers_tbd
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    layers, layers_tbd = get_layers(net)
    print '\n============  Surgery: threshold=%0.2f   ============' % threshold
    for i, layer in enumerate(layers_tbd):
        W = net.params[layer][0].data
        b = net.params[layer][1].data
        # hi = np.max(np.abs(W.flatten()))
        hi = np.std(W.flatten())
        local_threshold = threshold
        if layer == layers_tbd[0]:
            local_threshold = 0.1
        if layer == layers_tbd[1]:
            local_threshold = threshold - 0.2
        if layer == layers_tbd[2]:
            local_threshold = threshold
        if layer == layers_tbd[3]:
            local_threshold = threshold + 0.01
        if layer == layers_tbd[4]:
            local_threshold = threshold + 0.01

        mask = (np.abs(W) > (hi * local_threshold))
        mask = np.bool_(mask)
        W = W * mask
        print 'non-zero W percentage = %0.5f ' % (np.count_nonzero(W.flatten()) / float(np.prod(W.shape)))
        net.params[layer][0].data[...] = W
        # net.params[layer][0].mask[...] = mask
        print mask.shape

    (total_percentage, percentage_list) = analyze_param(net, layers)
    output_model = output_prefix + str(threshold) + '_' + suffix + ".caffemodel"
    '''
    if save_net == 1:
        net.save(output_model)
    '''
    return (threshold, total_percentage, percentage_list)


results = Parallel(n_jobs=num_cores)(delayed(prune)(threshold) for threshold in threshold_list)

for i, result in enumerate(results):
    print "result%d:"%i, result
for (threshold, total_percentage, percentage_list) in results:
    fout.write("%4.2f, %.5f,\n" % (threshold, total_percentage))

for (threshold, total_percentage, percentage_list) in results:
    fout2.write("%4.2f, %.5f, %.5f, %.5f, %.5f, %.5f\n" % (threshold, percentage_list[0], percentage_list[1], percentage_list[2], percentage_list[3], percentage_list[4]))

fout.close()
fout2.close()
sys.exit(0)

