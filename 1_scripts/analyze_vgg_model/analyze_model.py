'''
Created on Apr 18, 2015

@author: huizi, songhan
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

def analyze_param(net, layers, fout = None):
#   plt.figure()
    print '\n=============analyze_param start==============='
    total_nonzero = 0
    total_allparam = 0
    percentage_list = []
    all_layers = net.blobs.keys()

    # this_layer_percentage_D = 1.0 # Initialize the percentage of non-zero activations to 1.0

    # eval Data
    non_zero_D = np.zeros((len(layers)))
    all_param_D = np.zeros((len(layers)))
    iters = 1000
    for iter in range(iters):
        net.forward()
        print "Iter: ", iter

        for i, layer in enumerate(layers):
            D = net.blobs[all_layers[all_layers.index(layer)-1]].data

            non_zero_D[i] += np.count_nonzero(D.flatten())
            all_param_D[i] += np.prod(D.shape)

    non_zero_D /= iters * D.shape[0]
    all_param_D /= iters * D.shape[0]

    # eval all
    for i, layer in enumerate(layers):
        W = net.params[layer][0].data
        b = net.params[layer][1].data
        D = net.blobs[all_layers[all_layers.index(layer)-1]].data
        this_layer_D = net.blobs[layer].data
        print layer,
        print "kernel shape=", W.shape
        print "data   shape= ", D.shape

        # Eval weights
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

        print '%s range = [%f, %f]' % (layer, min(D.flatten()), max(D.flatten()))
        print '%s mean = %f, std = %f' % (layer, np.mean(D.flatten()), np.std(D.flatten()))

        '''
        # eval Last layer data
        non_zero_D_last = np.count_nonzero(D_last.flatten())
        all_param_D_last = np.prod(D_last.shape)
        this_layer_percentage_D_last = non_zero_D_last / float(all_param_D_last)
        '''

        # Eval flops
        this_layer_percentage_D = non_zero_D[i] / all_param_D[i]
        all_flops = 2 * W.size * this_layer_D.size / W.shape[0] / this_layer_D.shape[0]
        flops_percent = this_layer_percentage_D * this_layer_percentage
        non_zero_flops = int(all_flops * flops_percent)
        print 'Flops total = %d' % all_flops
        print 'Flops after pruning = %d' % non_zero_flops
        print 'percentage = %f\n' % flops_percent


        if fout is not None:
            fout.write('%s, %d, %d, %1.3f, %1.3f, %1.3f, ,%d , %d ,%d ,%d\n'%(layer, all_param, all_flops, this_layer_percentage_D, this_layer_percentage, flops_percent, non_zero, all_param_D[i], non_zero_D[i], non_zero_flops))

    print '=====> summary:'
    print 'non-zero W and b cnt = %d' % total_nonzero
    print 'total W and b cnt = %d' % total_allparam
    print 'percentage = %f' % (total_nonzero / float(total_allparam))
    print '=============analyze_param ends ==============='
    return (total_nonzero / float(total_allparam), percentage_list)

caffe.set_mode_gpu()
caffe.set_device(0)
prototxt = '3_prototxt_solver/vgg16/train_val.prototxt'
caffemodel = '4_model_checkpoint/vgg16/vgg16_12x.caffemodel'
xls_file = '2_results/VGG16/res.xls'

net = caffe.Net(prototxt, caffemodel, caffe.TEST)

layers = filter(lambda x:'conv' in x or 'fc' in x, net.params.keys())
fout = open(xls_file,'w')
analyze_param(net, layers, fout)
fout.close()

command = caffe_root + "/build/tools/caffe test --model=" + prototxt + " --weights=" + caffemodel + " --iterations=1000 --gpu 2 "
print command
os.system(command)

