'''
Created on Apr 18, 2015

@author: songhan
'''

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
    percentage_list = []
    for i, layer in enumerate(layers):
        i += 1
        W = net.params[layer][0].data
        b = net.params[layer][1].data
#       plt.subplot(3, 1, i);
#       numBins = 2 ^ 8
#       plt.hist(W.flatten(), numBins, color='blue', alpha=0.8)
#       plt.show()
        print layer,
        print "kernel shape=", W.shape

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


def analyze_data(net, layers):
    print '\n=============analyze_data start==============='
    for i, layer in enumerate(layers):
        D = net.blobs[layer].data
        print layer,
        print " data shape= ", D.shape
        print ' range = [%f, %f]' % (min(D.flatten()), max(D.flatten()))
        print ' mean = %f, std = %f' % (np.mean(D.flatten()), np.std(D.flatten()))
        non_zero = np.count_nonzero(D.flatten())
        all_param = np.prod(D.shape)
        this_layer_percentage = non_zero / float(all_param)
#         total_nonzero += non_zero
#         total_allparam += all_param
        print ' non-zero D cnt = %d' % non_zero
        print ' total D cnt = %d' % all_param
        print ' percentage = %f\n' % (this_layer_percentage)




folder = "/L2/"
prototxt = caffe_root + '/5_bac/' + 'train_val1.44.prototxt'
caffemodel = caffe_root + '/4_model_checkpoint/2_after_retrain/L2/' + "prune1.44_iter_800000.caffemodel"
# caffemodel = caffe_root + '/4_model_checkpoint/1_before_retrain/L2/' + "alex_pruned_1.44_678half.caffemodel"
if folder[2] == '1':
    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6_new', 'fc7_new', 'fc8_new']
if folder[2] == '2':
    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']

net = caffe.Net(prototxt, caffemodel, caffe.TEST)
net.forward()
layers = net.params.keys()
blobs = net.blobs.keys()

analyze_param(net, layers)
analyze_data(net, blobs)

# command = caffe_root + "/build/tools/caffe test --model=" + prototxt + " --weights=" + caffemodel + " --iterations=1000 --gpu 0"
# print command
# os.system(command)


