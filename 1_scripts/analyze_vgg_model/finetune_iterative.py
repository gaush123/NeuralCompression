import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.vq as scv
import pickle

# os.system("cd $CAFFE_ROOT")
caffe_root = os.environ["CAFFE_ROOT"]
os.chdir(caffe_root)
print caffe_root
sys.path.insert(0, caffe_root + 'python')
import caffe

device_id = int(sys.argv[1])
choice = [2 ** int(sys.argv[2]), 2 ** int(sys.argv[3])]

if len(sys.argv) > 4:
    train_opt = sys.argv[4]
else:
    train_opt = ""

caffe.set_mode_gpu()
caffe.set_device(device_id)
option = 'alexnet'
if option == 'lenet5':
    prototxt = '3_prototxt_solver/lenet5/train_val.prototxt'             
    caffemodel = '4_model_checkpoint/lenet5/lenet5.caffemodel'
    solver_proto = '3_prototxt_solver/lenet5/lenet_solver_finetune.prototxt'
    iters = 100
    dir_t = '2_results/kmeans/lenet5/'
    snap_dir = '4_model_checkpoint/lenet5/snapshot/'
elif option == 'alexnet':
    prototxt = '3_prototxt_solver/L2/train_val.prototxt'             
    solver_proto = '3_prototxt_solver/L2/finetune_solver.prototxt'
    caffemodel = '4_model_checkpoint/alexnet/alexnet9x.caffemodel'  
    iters = 1000
    dir_t = '2_results/kmeans/alexnet/'
    snap_dir = '4_model_checkpoint/alexnet/snapshot/'
elif option == 'vgg':
    prototxt = '3_prototxt_solver/vgg16/train_val.prototxt'             
    caffemodel = '4_model_checkpoint/vgg16/vgg16_12x.caffemodel'  
    iters = 1000
    dir_t = '2_results/kmeans/vgg16/'

log = dir_t + 'log_accu'

solver = caffe.SGDSolver(solver_proto)
solver.net.copy_from(caffemodel)
net = solver.net

layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())
if option == 'lenet5':
    num_c = [choice[0]] * (len(layers) - 2) + [choice[1]] * 2
else:
    num_c = [choice[0]] * (len(layers) - 3) + [choice[1]] * 3

print "layers TBD: ", layers
print "num_c = ", num_c

print "==============1 Perform K-means============="
codebook = {}
for idx, layer in enumerate(layers):
    print "Eval layer:", layer
    W = net.params[layer][0].data.flatten()
    W = W[np.where(W != 0)]
    std = np.std(W)
    initial_uni = np.linspace(-4 * std, 4 * std, num_c[idx]-1)
    codebook[layer],_= scv.kmeans(W, initial_uni)    
    codebook[layer] = np.append(0.0, codebook[layer])
    print "codebook size:", len(codebook[layer])

print "================2 Perform quantization=============="

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
    codeDict={}
    maskCode={}
    for layer in layers:
        print "Quantize layer:", layer
        W = net.params[layer][0].data
        if use_stochastic:
            codes = stochasitc_quantize2(W.flatten(), codebook[layer]) 
        else:
            codes, _ = scv.vq(W.flatten(), codebook[layer])
        W_q = np.reshape(codebook[layer][codes], W.shape)
        net.params[layer][0].data[...] = W_q

        maskCode[layer] = np.reshape(codes, W.shape)
        codeBookSize = len(codebook[layer])    
        a = maskCode[layer].flatten()
        b = xrange(len(a))

        codeDict[layer]={}
        for i in xrange(len(a)):
            codeDict[layer].setdefault(a[i], []).append(b[i])

    return codeDict, maskCode

codeDict, maskCode = quantize_net(net, codebook)

print "================3 Perform fintuning=============="
extra_lr=5e-6
import time
decay_rate = 0.99 
momentum=0.9
update=train_opt
if update=='':
    update = 'sgd'
if update=='adadelta':
    step_cache2 = {}
    for layer in layers:
        step_cache2[layer] = {}
        for code in xrange(1, len(codebook[layer])):
            step_cache2[layer][code] = 0.0
    smooth_eps = 1e-8


start_time=time.time()
step_cache = {}
for layer in layers:
    step_cache[layer] = {}
    for code in xrange(1, len(codebook[layer])):
        step_cache[layer][code] = 0.0

for i in xrange(3000):
    if (i > 0 and i % 200 == 0):
        net.copy_from(caffemodel)
        codeDict, maskCode = quantize_net(net, codebook, True)
    solver.step(1)
    for layer in layers:
        diff=net.params[layer][0].diff.flatten()
        codeBookSize=len(codebook[layer])
        for code in xrange(1,codeBookSize):
            indexes = codeDict[layer][code]
            diff_ave=np.sum(diff[indexes])/len(indexes)
            if update == 'sgd':
                dx = -extra_lr * diff_ave
            elif update == 'momentum':
                dx = momentum * step_cache[layer][code] - (1-momentum) * extra_lr * diff_ave
                step_cache[layer][code] = dx                
            elif update == 'rmsprop':
                step_cache[layer][code] =  decay_rate * step_cache[layer][code] + (1.0 - decay_rate) * diff_ave ** 2
                dx = -(extra_lr* diff_ave) / np.sqrt(step_cache[layer][code] + 1e-6)
            elif update == 'adadelta':                                                                              
                step_cache[layer][code] = step_cache[layer][code] * decay_rate + (1.0 - decay_rate) * diff_ave ** 2           
                dx = - np.sqrt( (step_cache2[layer][code] + smooth_eps) / (step_cache[layer][code] + smooth_eps) ) * diff_ave
                step_cache2[layer][code] = step_cache2[layer][code] * decay_rate + (1.0 - decay_rate) * (dx ** 2)             

            codebook[layer][code] += dx
        W2 = codebook[layer][maskCode[layer]]
        
        net.params[layer][0].data[...]=W2
    # print np.std(net.params[layer][0].diff)
