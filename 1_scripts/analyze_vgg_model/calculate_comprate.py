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

def analyze_comprate(net, layers, params_bits, location_bits = None, encoding='pack'):
    assert len(layers) == len(params_bits)
    if encoding == 'pack':
        assert len(location_bits) == len(layers)
        num_ori = 0
        num_new = 0
        total_extra_slots = 0
        total_non_zeros = 0
        # Parameters and locations bits
        for idx, layer in enumerate(layers):
            num_ori += net.params[layer][0].data.size * 32
            num_ori += net.params[layer][1].data.size * 32
            max_length = 2 ** location_bits[idx]
            non_zeros_loc = np.where(net.params[layer][0].data.flatten() != 0.0)[0]
            distance_loc = non_zeros_loc[1:] - non_zeros_loc[:-1]
            extra_slots = np.sum(np.floor(distance_loc / max_length))
            total_extra_slots += extra_slots
            non_zeros = np.count_nonzero(net.params[layer][0].data)
            total_non_zeros += non_zeros
            num_new +=(non_zeros+ extra_slots) * (params_bits[idx] + location_bits[idx])
            num_new += net.params[layer][1].data.size * 32
            print "Layer:", layer
            print "Extra slots:",extra_slots
            print "Non-zeros:", non_zeros
            print "Extra slots rate", float(extra_slots) / non_zeros
            print "====================================="


        print "total extra slots"
        print total_extra_slots
        print "total non-zeros"
        print total_non_zeros

        # Codebooks
        ndlist = np.array(bits_list)
        num_new += 32 * np.sum(2 ** ndlist)
        return float(num_new) / num_ori
    else:
        raise Exception("Unsupported encoding method!")


caffe.set_mode_gpu()
caffe.set_device(0)
prototxt = '3_prototxt_solver/L2/train_val.prototxt'             
caffemodel = '4_model_checkpoint/alexnet/alexnet9x.caffemodel'  

net = caffe.Net(prototxt, caffemodel, caffe.TEST)

layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())
bits_list = [8] * (len(layers) - 3) + [4] * 3
location_bits = [5] * len(layers)

rate = analyze_comprate(net, layers, bits_list, location_bits)

print "===================Results====================="
print "Compression rate:", rate

