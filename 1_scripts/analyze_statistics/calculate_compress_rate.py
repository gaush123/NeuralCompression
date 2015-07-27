'''
Created on Apr 18, 2015

@author: huizi, songhan
'''

import sys
import os
import numpy as np


# os.system("cd $CAFFE_ROOT")
caffe_root = os.environ["CAFFE_ROOT"]
sys.path.insert(0, caffe_root + 'python')
os.chdir(caffe_root)
import caffe

def analyze_comprate(net, layers, params_bits, location_bits=None, encoding='pack', fout=None):
    assert len(layers) == len(params_bits)
    if encoding == 'pack':
        assert len(location_bits) == len(layers)
        num_ori = 0
        num_new = 0
        total_extra_slots = 0
        total_non_zeros = 0
        total_param_num = 0
        # Parameters and locations bits
        for idx, layer in enumerate(layers):
            num_ori += net.params[layer][0].data.size * 32
            num_ori += net.params[layer][1].data.size * 32
            max_length = 2 ** location_bits[idx]

            non_zeros_loc = np.where(net.params[layer][0].data.flatten() != 0.0)[0]
            distance_loc = non_zeros_loc[1:] - non_zeros_loc[:-1]
            extra_slots = np.sum(np.floor(distance_loc / max_length))
            non_zeros = np.count_nonzero(net.params[layer][0].data)

            total_extra_slots += extra_slots
            total_non_zeros += non_zeros
            num_new += (non_zeros + extra_slots) * (params_bits[idx] + location_bits[idx])
            total_param_num += (non_zeros + extra_slots) * params_bits[idx]
            num_new += net.params[layer][1].data.size * 32

            print "Layer:", layer
            print "Extra slots:", extra_slots
            print "Non-zeros:", non_zeros
            print "Extra slots rate", float(extra_slots) / non_zeros
            print "====================================="


        print "total extra slots ratio:"
        extra_ratio = float(total_extra_slots) / (total_non_zeros + total_extra_slots)
        print extra_ratio

        # Codebooks
        ndlist = np.array(params_bits)
        codebook_size = 32 * np.sum(2 ** ndlist)
        num_new += codebook_size

        comp_rate = float(num_new) / num_ori
        codebook_ratio = float(codebook_size) / num_new
        params_ratio = float(total_param_num) / num_new
        loc_ratio = 1 - codebook_ratio - params_ratio
        if fout is not None:
            fout.write('%f, %f, %f, %f, %f\n' % (comp_rate, codebook_ratio, params_ratio, loc_ratio, extra_ratio))

        return comp_rate
    else:
        raise Exception("Unsupported encoding method!")


caffe.set_mode_gpu()
caffe.set_device(1)
option = 'alexnet'
if option == 'lenet5':
    prototxt = '3_prototxt_solver/lenet5/train_val.prototxt'
    caffemodel = '4_model_checkpoint/lenet5/lenet5.caffemodel'
    solver_proto = '3_prototxt_solver/lenet5/lenet_solver_finetune.prototxt'
elif option == 'alexnet':
    prototxt = '3_prototxt_solver/L2/train_val.prototxt'
    solver_proto = '3_prototxt_solver/L2/finetune_solver.prototxt'
    caffemodel = '4_model_checkpoint/alexnet/alexnet9x.caffemodel'
elif option == 'vgg':
    prototxt = '3_prototxt_solver/vgg16/train_val.prototxt'
    caffemodel = '4_model_checkpoint/vgg16/vgg16_12x.caffemodel'
    solver_proto = '3_prototxt_solver/L2/finetune_solver.prototxt'
    snap_dir = '4_model_checkpoint/4_model_checkpoint/vgg16/snapshot/'

net = caffe.Net(prototxt, caffemodel, caffe.TEST)

layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())
def main(choice=[4, 3, 5], fout=None):
    if option == 'lenet5':
        bits_list = [choice[0]] * (len(layers) - 2) + [choice[1]] * 2
    else:
        bits_list = [choice[0]] * (len(layers) - 3) + [choice[1]] * 3
    location_bits = [choice[2]] * len(layers)

    rate = analyze_comprate(net, layers, bits_list, location_bits, fout=fout)

    print "===================Results====================="
    print "Compression rate:", rate

def get_results(choice=[4, 3], file_out=''):
    if option == 'lenet5':
        bits_list = [choice[0]] * (len(layers) - 2) + [choice[1]] * 2
    else:
        bits_list = [choice[0]] * (len(layers) - 3) + [choice[1]] * 3

    rate_min = 1.0
    for location_bits in range(1, 9):
        rate = analyze_comprate(net, layers, bits_list, [location_bits] * len(layers))
        if rate < rate_min:
            rate_min = rate
            optimal_loc_bits = location_bits

    if file_out == '':
        return optimal_loc_bits, rate_min
    with open(file_out, 'a+') as f:
        f.write('%d, %f\n' % (optimal_loc_bits, rate_min))
    return

if __name__ == "__main__":
    # print get_results(choice = map(lambda x:int(x), sys.argv[1:3]))
    log_file = 'compress_lenet'
    get_results([5, 3], log_file)
    get_results([4, 3], log_file)
    get_results([4, 2], log_file)
    get_results([8, 5], log_file)
    get_results([8, 6], log_file)
    get_results([6, 4], log_file)
    get_results([6, 3], log_file)
    get_results([5, 4], log_file)
