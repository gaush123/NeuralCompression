
import sys
import os
import numpy as np
import pickle

help_ = '''
Usage:
    decode.py <net.prototxt> <net.binary> <target.caffemodel>
    Set variable CAFFE_ROOT as root of caffe before run this demo!
'''

if len(sys.argv) != 4:
    print help_
    sys.exit()
else:
    prototxt = sys.argv[1]
    net_bin = sys.argv[2]
    target = sys.argv[3]

# os.system("cd $CAFFE_ROOT")
caffe_root = os.environ["CAFFE_ROOT"]
os.chdir(caffe_root)
print caffe_root
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_cpu()
net = caffe.Net(prototxt, caffe.TEST)
layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())

fin = open(net_bin, 'rb')

def binary_to_net(weights, spm_stream, ind_stream, codebook, num_nz):
    bits = np.log2(codebook.size)
    if bits == 4:
        slots = 2
    elif bits == 8:
        slots = 1
    else:
        print "Not impemented,", bits
        sys.exit()
    code = np.zeros(weights.size, np.uint8) 

    # Recover from binary stream
    spm = np.zeros(num_nz, np.uint8)
    ind = np.zeros(num_nz, np.uint8)
    if slots == 2:
        spm[np.arange(0, num_nz, 2)] = spm_stream % (2**4)
        spm[np.arange(1, num_nz, 2)] = spm_stream / (2**4)
    else:
        spm = spm_stream
    ind[np.arange(0, num_nz, 2)] = ind_stream% (2**4)
    ind[np.arange(1, num_nz, 2)] = ind_stream/ (2**4)


    # Recover the matrix
    ind = np.cumsum(ind+1)-1
    code[ind] = spm
    data = np.reshape(codebook[code], weights.shape)
    np.copyto(weights, data)

nz_num = np.fromfile(fin, dtype = np.uint32, count = len(layers))
for idx, layer in enumerate(layers):
    print "Reconstruct layer", layer
    print "Total Non-zero number:", nz_num[idx]
    if 'conv' in layer:
        bits = 8
    else:
        bits = 4
    codebook_size = 2 ** bits
    codebook = np.fromfile(fin, dtype = np.float32, count = codebook_size)
    bias = np.fromfile(fin, dtype = np.float32, count = net.params[layer][1].data.size)
    np.copyto(net.params[layer][1].data, bias)

    spm_stream = np.fromfile(fin, dtype = np.uint8, count = (nz_num[idx]-1) / (8/bits) + 1)
    ind_stream = np.fromfile(fin, dtype = np.uint8, count = (nz_num[idx]-1) / 2+1)

    binary_to_net(net.params[layer][0].data, spm_stream, ind_stream, codebook, nz_num[idx])

net.save(target)
