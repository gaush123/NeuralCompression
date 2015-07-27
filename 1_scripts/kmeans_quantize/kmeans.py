'''
Created on Apr 18, 2015

@author: huizi, songhan
'''

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.vq_maohz as scv
import pickle

# os.system("cd $CAFFE_ROOT")
caffe_root = os.environ["CAFFE_ROOT"]
os.chdir(caffe_root)
print caffe_root
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()
caffe.set_device(2)
option = 'alexnet'
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

log = dir_t + 'log_accu'

def kmeans_net(net, layers, num_c = 16, initials=None):
    codebook = {}
    if type(num_c) == type(1):
        num_c = [num_c] * len(layers)
    else:
        assert len(num_c) == len(layers)

    print "==============Perform K-means============="
    for idx, layer in enumerate(layers):
        print "Eval layer:", layer
        W = net.params[layer][0].data.flatten()
        W = W[np.where(W != 0)]
        if initials is None: #Default: uniform sample
            min_W = np.min(W)
            max_W = np.max(W)
            initial_uni = np.linspace(min_W, max_W, num_c[idx] - 1)
            codebook[layer],_= scv.kmeans(W, initial_uni, compress=False)
            '''
            codebook[layer],_= scv.kmeans(W, num_c[idx] - 1)
            '''
        elif type(initials) == type(np.array([])):
            codebook[layer],_= scv.kmeans(W, initials)
        else:
            print type(initials)
            return None
        codebook[layer] = np.append(0.0, codebook[layer])
        print "codebook size:", len(codebook[layer])
    return codebook
def stochasitc_quantize2(W, codebook):
    
    mask = W[:,np.newaxis] - codebook
    
    mask_neg = mask
    mask_neg[mask_neg>0.0] -= 99999.0
    max_neg = np.max(mask_neg, axis=1)
    max_code = np.argmax(mask_neg, axis = 1)
    
    mask_pos = mask
    mask_pos += 99999.0
    min_code = np.argmin(mask_pos, axis = 1)
    min_pos = np.min(mask_pos, axis=1)
 
    rd = np.random.uniform(low=0.0, high=1.0, size=(len(W)))
    thresh = min_pos.astype(np.float32)/(min_pos - max_neg)
    
    max_idx = thresh < rd
    min_idx = thresh >= rd

    codes = np.zeros(W.shape)
    codes[max_idx] += min_code[max_idx]
    codes[min_idx] += max_code[min_idx]
    
    return codes.astype(np.int)

def quantize_net(net, codebook, use_stochastic=False):
    layers = codebook.keys()
    print "================Perform quantization=============="
    for layer in layers:
        print "Quantize layer:", layer
        W = net.params[layer][0].data
        if use_stochastic:
            codes = stochasitc_quantize2(W.flatten(), codebook[layer]) 
        else:
            codes, _ = scv.vq(W.flatten(), codebook[layer])
        W_q = np.reshape(codebook[layer][codes], W.shape)
        np.copyto(net.params[layer][0].data, W_q)

def parse_caffe_log(log):
    lines = open(log).readlines()
    return map(lambda x: float(x.split()[-1]), lines[-3:-1])


def main(choice = [64,16] ):
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
# layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())
    layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())

# Evaluate the origina accuracy
    command = caffe_root + "/build/tools/caffe test --model=" + prototxt + " --weights=" + caffemodel + " --iterations=%d --gpu 1 2>"%iters +log
    #print command
    # os.system(command)
    #os.system('tail -n 3 '+ log)

    num_c = [choice[0]]  * (len(layers)-3) + [choice[1]] * 3
    codebook = kmeans_net(net, layers, num_c)

    pickle.dump(codebook, open(dir_t + 'codebook.pkl', 'w'))
    quantize_net(net, codebook)

# Evaluate the new model's accuracy
    '''
    net.save(caffemodel + '.quantize')
    command = caffe_root + "/build/tools/caffe test --model=" + prototxt + " --weights=" + caffemodel + ".quantize --iterations=%d --gpu 1 2>"%iters +log + "new"
    #print command
    print choice
    os.system(command)
    os.system('tail -n 3 '+ log + 'new')

    top_1,top_5 = parse_caffe_log(log + 'new')
    with open('results_%s'%option,'a+') as f:
        f.write('%d %d \n%f\n%f\n'%(choice[0], choice[1], top_1, top_5))
    '''



def main2():
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
# layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())
    layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())

# Evaluate the origina accuracy
    command = caffe_root + "/build/tools/caffe test --model=" + prototxt + " --weights=" + caffemodel + " --iterations=%d --gpu 1 2>"%iters +log
    print command
    # os.system(command)
    os.system('tail -n 3 '+ log)

    bits = [8,8,7,6,6,3,3,4]
    num_c = 2 ** bits 
    codebook = kmeans_net(net, layers, num_c)

    pickle.dump(codebook, open(dir_t + 'codebook.pkl', 'w'))

    quantize_net(net, codebook)

# Evaluate the new model's accuracy
    net.save(caffemodel + '.quantize')
    command = caffe_root + "/build/tools/caffe test --model=" + prototxt + " --weights=" + caffemodel + ".quantize --iterations=%d --gpu 1 2>"%iters +log + "new"
    print command
    os.system(command)
    os.system('tail -n 3 '+ log + 'new')

if __name__ == "__main__":
    '''
    main([256,8])
    main([64,8])
    main([64,16])
    main([256,16])
    main2()
    '''
    main([64,16])
