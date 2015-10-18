
import sys
import os
import numpy as np
import pickle

# os.system("cd $CAFFE_ROOT")
caffe_root = os.environ["CAFFE_ROOT"]
os.chdir(caffe_root)
print caffe_root
sys.path.insert(0, caffe_root + 'python')
import caffe


def to_binary(array, types):
    bits = int(np.log2(types-1))+1
    if bits == 4:
        slots = 2
    elif bits == 8:
        slots = 1
    else:
        print "Not impemented,", bits
        sys.exit()
    stream_len =(len(array) -1)/slots+1
    stream = np.zeros(stream_len, np.uint8)
    for i in range(slots):
        data = array[np.arange(i, len(array), slots)]
        stream[:len(data)] += data * (2**(bits*i))

    return stream


caffe.set_mode_cpu()
option = 'vgg'
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


caffemodel = caffe_root + caffemodel + '.quantize'
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())

codebook = pickle.load(open(caffe_root + 'codebook.pkl'))
codes_W = pickle.load(open(caffe_root + 'codes.pkl'))

fout = open('compress.net','wb')
max_jump = 16
nz_num = np.zeros(len(layers), np.uint32)
spm_stream = [0] * len(layers)
ind_stream = [0] * len(layers)
for idx, layer in enumerate(layers):
    W = codes_W[layer].flatten()
    spm_tmp = np.zeros(W.size, dtype = np.uint32)
    ind_tmp = np.ones(W.size, dtype = np.uint32) * (max_jump-1)
    loc = np.where(W!=0)[0]
    distance_loc = np.append(loc[0], np.diff(loc)-1)  #jump 1 encode to 0
    zeros = distance_loc/max_jump
    idx_vec = np.cumsum(zeros+1)-1  #add the element itself. first one need -1
    total_slot = idx_vec[-1]+1
    nz_num[idx] = total_slot
    spm_tmp[idx_vec] = W[loc]
    ind_tmp[idx_vec] = distance_loc % max_jump
    print layer
    if idx == 0:
        print ind_tmp[:40]

    spm_stream[idx] = to_binary(spm_tmp[:total_slot], codebook[layer].size)
    ind_stream[idx] = to_binary(ind_tmp[:total_slot], max_jump)

nz_num.tofile(fout)
for idx, layer in enumerate(layers):
    codebook[layer].astype(np.float32).tofile(fout)
    net.params[layer][1].data.tofile(fout)
    spm_stream[idx].tofile(fout)
    ind_stream[idx].tofile(fout)
fout.close()
