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
            max_length = 2 ** location_bits[idx] - 1

            non_zeros_loc = np.where(net.params[layer][0].data.flatten() != 0.0)[0]
            distance_loc = non_zeros_loc[1:] - non_zeros_loc[:-1]
            extra_slots = np.sum(np.floor(distance_loc / max_length))
            non_zeros = np.count_nonzero(net.params[layer][0].data)

            total_extra_slots += extra_slots
            total_non_zeros += non_zeros
            num_new += (non_zeros ) * (params_bits[idx] + location_bits[idx]) + extra_slots * location_bits[idx]
            total_param_num += (non_zeros ) * params_bits[idx]
            num_new += net.params[layer][1].data.size * 32

        extra_ratio = float(total_extra_slots) / (total_non_zeros + total_extra_slots)
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

        return comp_rate, codebook_ratio, params_ratio
    else:
        raise Exception("Unsupported encoding method!")


caffe.set_mode_gpu()
caffe.set_device(1)
option = 'vgg'
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
    caffemodel = '4_model_checkpoint/vgg16/vgg16_13x.caffemodel'
    solver_proto = '3_prototxt_solver/L2/finetune_solver.prototxt'
    snap_dir = '4_model_checkpoint/4_model_checkpoint/vgg16/snapshot/'
elif option == 'lenet_300':
    prototxt = '3_prototxt_solver/lenet_300_100/train_val.prototxt'
    caffemodel = '4_model_checkpoint/lenet_300_100/lenet300_100_9x.caffemodel'
    solver_proto = '3_prototxt_solver/lenet_300_100/finetune_solver.prototxt'

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
    if len(choice) == 2:
        if option == 'lenet5':
            bits_list = [choice[0]] * (len(layers) - 2) + [choice[1]] * 2
        elif option == 'lenet_300':
            bits_list = [choice[1]] * len(layers)
        else:
            bits_list = [choice[0]] * (len(layers) - 3) + [choice[1]] * 3
    else:
        assert len(choice) == len(layers)
        bits_list = choice

    rate_min = 1.0
    for location_bits in range(4, 7):
        rate, a, b = analyze_comprate(net, layers, bits_list, [location_bits] * len(layers))
        if rate < rate_min:
            rate_min = rate
            optimal_loc_bits = location_bits
            codebook_ratio = a
            params_ratio  = b

        print "rate:", rate
    if file_out == '':
        return optimal_loc_bits, rate_min
    with open(file_out, 'a+') as f:
        f.write('%d, %f, %f, %f\n' % (optimal_loc_bits, rate_min, codebook_ratio, params_ratio))
    return



def test_alexnet_only_pruned():
    setting = [32,32]
    log_file = 'tmp/only_pruned'
    get_results(setting, log_file)

def calculate_rate(setting):
    if len(setting) == 2:
        setting_new = [setting[0]] * 5 + [setting[1]] * 3
    else:
        setting_new = setting
    total_num = 0
    new_num = 0
    for idx, layer in enumerate(layers):
        new_num += net.params[layer][0].data.size * setting_new[idx]
        total_num += net.params[layer][0].data.size * 32

    return float(new_num) / total_num

def test_alexnet_only_quanzied():
    settings = [[6  ,3  ],
    [7  ,3  ],
    [9  ,3  ],
    [11 ,3  ],
    [11 ,2  ],
    [8  ,2  ],
    [5  ,2  ],
    [4  ,2  ],
    [10,8, 8, 8, 8, 2, 3, 2    ],
    [6 ,6, 6, 10, 10, 2, 3, 3  ],
    [10,10, 8, 10, 10, 3, 2, 3 ],
    [8 ,8, 8, 6 ,6  ,2 ,2 ,3   ],
    [8 ,8, 8, 6 ,5  ,2 ,2 ,3   ],
    [10,10, 8, 8 ,6  ,2 ,2 ,3   ],
    [8 ,8, 6, 6 ,5  ,2 ,2 ,3   ]]
    log_file = 'tmp/only_quantized'
    with open(log_file, 'w') as f:
        for setting in settings:
            rate = calculate_rate(setting)
            f.write('%f\n'%rate)

def test_alexnet():
    # print get_results(choice = map(lambda x:int(x), sys.argv[1:3]))
    log_file = 'tmp/compress_alexnet'
    settings = [[8  ,4  ],
    [9  ,4  ],
    [10 ,4  ],
    [11 ,4  ],
    [8  ,5  ],
    [9  ,5  ],
    [10 ,5  ],
    [11 ,5  ],
    [9  ,6  ],
    [10 ,6  ],
    [11 ,6  ],
    [8  ,7  ],
    [9  ,7  ],
    [10 ,7  ],
    [11 ,7  ],
    [8  ,8  ],
    [9  ,8  ],
    [10 ,8  ],
    [11 ,8  ],
    [9  ,9  ],
    [10 ,9  ],
    [11 ,9  ],
    [10 ,10 ],
    [11 ,10 ],
    [7  ,3  ],
    [8  ,3  ],
    [9  ,3  ],
    [10 ,3  ],
    [11 ,3  ],
    [5  ,2  ],
    [4  ,2  ],
    [3  ,2  ],
    [2  ,2  ],
    [6 ,6, 6, 6 ,6  ,2 ,2 ,2  ],        
    [6 ,6, 6, 10, 10, 2, 3, 3 ],    
    [8 ,8, 8, 6 ,6  ,2 ,2 ,3     ],
    [8 ,8, 8, 6 ,5  ,2 ,2 ,3     ],
    [8 ,8, 6, 6 ,5  ,2 ,2 ,3    ]]

    for setting in settings:
        get_results(setting, log_file)

def test_vgg():
    log_file = 'tmp/compress_vgg'
    settings = [[5,3],
    [4  ,3],
    [4  ,2],
    [8  ,5],
    [8  ,6],
    [6  ,4],
    [6  ,3],
    [5  ,4],
    [8  ,4]]
    for setting in settings:
        get_results(setting, log_file)

def test_lenet():
    log_file = 'tmp/compress_lenet'
    settings = [[5,3],
    [4  ,3],
    [4  ,2],
    [8  ,5],
    [8  ,6],
    [6  ,4],
    [6  ,3],
    [5  ,4]]
    for setting in settings:
        get_results(setting, log_file)


def test_lenet_300():
    log_file = 'tmp/compress_lenet'
    settings = [[5,10],
    [4  ,8],
    [4  ,6],
    [8  ,4],
    [8  ,3],
    [6  ,2]]
    for setting in settings:
        get_results(setting, log_file)


if __name__ == "__main__":
    test_vgg()
