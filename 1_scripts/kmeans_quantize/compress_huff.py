
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
    caffemodel = '4_model_checkpoint/vgg16/vgg16_13x.caffemodel'
    iters = 1000
    dir_t = '2_results/kmeans/vgg16/'


caffemodel = caffe_root + caffemodel + '.quantize'
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())

codebook = pickle.load(open(caffe_root + 'codebook.pkl'))
codes_W = pickle.load(open(caffe_root + 'codes.pkl'))

def GetProbability(array):
    m = np.max(array)
    probs = map(lambda x:np.count_nonzero(array == x), range(m+1))
    probs = map(lambda x:float(x) / len(array), probs)
    return probs

def HuffmanEncode(probs):
    num = len(probs)
    codes = [0] * num
    code_lengths = [0] * num
    sub_groups = map(lambda x:[x], range(num))
    sub_groups_prob = list(probs)

    def find_min2(array):
        min1 = min(array)
        idx1 = array.index(min1)
        array[idx1] = 1.0
        min2 = min(array)
        idx2 = array.index(min2)
        array[idx1] = min1

        return idx1, idx2

    def append_codes(bit, idxs, codes, code_lengths):
        for idx in idxs:
            codes[idx] = codes[idx] * 2 + bit
            code_lengths[idx] += 1

    for i in range(num - 1):
        min_1, min_2 = find_min2(sub_groups_prob)

        append_codes(0, sub_groups[min_1], codes, code_lengths)
        append_codes(1, sub_groups[min_2], codes, code_lengths)

        sub_groups[min_1] = sub_groups[min_1] + sub_groups[min_2]
        del sub_groups[min_2]

        sub_groups_prob[min_1] = sub_groups_prob[min_1] + sub_groups_prob[min_2]
        del sub_groups_prob[min_2]


    return codes, code_lengths 

def ArrayToCode(array, codes, code_lengths):
    code_lengths = np.array(code_lengths, dtype = np.int32)
    total_length = np.sum(code_lengths[array])
    total_4bytes = (total_length - 1) / 32 + 1

    compressed_codes= np.zeros(total_4bytes, dtype = np.uint32)

    idx = 0
    shift = 0
    for i in range(len(array)):
        number = array[i]
        length = code_lengths[number]
        code = codes[number]
        while length > 0:
            eff = min(length, 32-shift)
            bits_to_write = code & ((1 << eff) - 1)
            if idx == total_4bytes:
                print i
                import IPython
                IPython.embed()
            compressed_codes[idx] += bits_to_write << shift

            code = code >> eff
            length -= eff
            idx += (shift + eff) / 32
            shift = (shift + eff) % 32

    assert idx * 32 + shift == total_length  # For debug

    return compressed_codes, total_length

def Decode(compressed_codes, total_length, codes, code_lengths, original_length):
    array = np.zeros(original_length, dtype = np.uint8)

    key = 0
    length = 0
    num = 0
    huff_quick_loc = map(lambda x: codes[x] + 2 ** code_lengths[x],
        range(len(codes)))
    for i in range(total_length):
        idx = i / 32
        shift = i % 32
        bit = (compressed_codes[idx] >> shift) & 0x1
        key += (1 << length) * bit
        length += 1
        quick_loc = key + (1 << length)
        if quick_loc in huff_quick_loc:
            array[num] = huff_quick_loc.index(quick_loc)
            num += 1
            key = 0
            length = 0

    assert num == original_length # For debug

    return array
        

ind_codes = 4
max_jump = 2 ** ind_codes
nz_num = np.zeros(len(layers), np.uint32)
spm = []
ind = []
for idx, layer in enumerate(layers):
    print "Deal with layer:", layer
    W = codes_W[layer].flatten()
    spm_tmp = np.zeros(W.size, dtype = np.uint16)
    ind_tmp = np.ones(W.size, dtype = np.uint16) * (max_jump-1)
    loc = np.where(W!=0)[0]
    distance_loc = np.append(loc[0], np.diff(loc)-1)  #jump 1 encode to 0
    zeros = distance_loc/max_jump
    idx_vec = np.cumsum(zeros+1)-1  #add the element itself. first one need -1
    total_slot = idx_vec[-1]+1
    nz_num[idx] = total_slot
    spm_tmp[idx_vec] = W[loc]
    ind_tmp[idx_vec] = distance_loc % max_jump

    spm.append(spm_tmp[:total_slot])
    ind.append(ind_tmp[:total_slot])

    probs = GetProbability(spm[idx])
    codes, code_lengths = HuffmanEncode(probs)
    compressed_codes, total_length = ArrayToCode(spm[idx], codes, code_lengths)
    recovered_array = Decode(compressed_codes, total_length, codes, code_lengths, len(spm[idx]))

    assert np.sum(recovered_array - spm[idx]) == 0
    sys.exit(0)

'''
nz_num.tofile(fout)
for idx, layer in enumerate(layers):
    codebook[layer].tofile(fout)
    net.params[layer][1].data.tofile(fout)
    spm[idx].tofile(fout)
    ind[idx].tofile(fout)
fout.close()
'''
