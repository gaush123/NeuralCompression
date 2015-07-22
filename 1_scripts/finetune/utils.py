import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.vq as scv
import pickle
import argparse
import time

parser = argparse.ArgumentParser()
#===========Basic options=============
parser.add_argument('--device-id', '-d', dest='device_id', type=int, default=0)
parser.add_argument('bits', type=int, nargs='+')
parser.add_argument('--layers', dest='layers', type=str, default='all')
parser.add_argument('--network', dest='network', type=str, default='alexnet')
parser.add_argument('--timing', dest='timing', type=bool, default=False)

#=======Training options===============
parser.add_argument('--lr', dest='lr', type=float, default=50)
parser.add_argument('--decay-rate', dest='decay_rate', type=float, default=0.99)
parser.add_argument('--momentum', dest='momentum', type=float, default=0.0)  
parser.add_argument('--update', dest='update', type=str, default='sgd')  

#===========For iterative training==============
parser.add_argument('--finetune-codebook-iters', dest='co_iters', type=int, default=1)  
parser.add_argument('--accumulate-diff-iters', dest='ac_iters', type=int, default=10)  
parser.add_argument('--stochastic', dest='use_stochastic', type=bool, default=True)  

args = parser.parse_args()

#=================Initializa Caffe==============
caffe_root = os.environ["CAFFE_ROOT"]
os.chdir(caffe_root)
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_gpu()        
caffe.set_device(args.device_id) 

#==============Set paths========================
option = args.network
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

#==================Initializa solver and net==========
solver = caffe.SGDSolver(solver_proto)
solver.net.copy_from(caffemodel)
net = solver.net

total_layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())
if args.layers == 'all':
    update_layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())
else:
    update_layers = filter(lambda x:args.layers in x, net.params.keys())

if len(args.bits) == 1:
    num_c = args.bits * len(total_layers)
elif len(args.bits) == 2:
    num_conv = len(filter(lambda x:'conv' in x, total_layers))
    num_c = [args.bits[0]] * (num_conv) + [args.bits[1]] * (len(total_layers) - num_conv)
else:
    num_c = args.bits
    assert len(num_c) == len(total_layers)
num_c = map(lambda x: 2 ** x, num_c)


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
            std = np.std(W)                                             
            initial_uni = np.linspace(-4 * std, 4 * std, num_c[idx] - 1)
            codebook[layer],_= scv.kmeans(W, initial_uni)               
        else:
            codebook[layer],_= scv.kmeans(W, initials)                  
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

def quantize_net_with_dict(net, layers, codebook, use_stochastic=False, timing=False):
    start_time = time.time()
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

    if args.timing:
        print "Update codebook time:%f"%(time.time() - start_time)

    return codeDict, maskCode

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(step_cache={}, step_cache2={}, initial=False)
def update_codebook_net(net, codebook, codeDict, maskCode, args, update_layers=None):

    start_time = time.time()
    extra_lr=args.lr
    decay_rate = args.decay_rate 
    momentum= args.momentum
    update_method= args.update

    if update_method == 'rmsprop':
        extra_lr /= 100

    if not update_codebook_net.initial:
        step_cache2 = update_codebook_net.step_cache2
        step_cache = update_codebook_net.step_cache
        if update_method=='adadelta':
            for layer in update_layers:
                step_cache2[layer] = {}
                for code in xrange(1, len(codebook[layer])):
                    step_cache2[layer][code] = 0.0
            smooth_eps = 1e-8

        for layer in update_layers:
            step_cache[layer] = {}
            for code in xrange(1, len(codebook[layer])):
                step_cache[layer][code] = 0.0
        
        update_codebook_net.initial = True
    else:
        step_cache2 = update_codebook_net.step_cache2
        step_cache = update_codebook_net.step_cache


    total_layers = net.params.keys()
    if update_layers is None:
        update_layers = total_layers

    for layer in total_layers:
        if layer in update_layers:
            diff=net.params[layer][0].diff.flatten()
            codeBookSize=len(codebook[layer])
            for code in xrange(1,codeBookSize):
                indexes = codeDict[layer][code]
                diff_ave=np.sum(diff[indexes])/len(indexes)
                if update_method == 'sgd':
                    dx = -extra_lr * diff_ave
                elif update_method == 'momentum':
                    dx = momentum * step_cache[layer][code] - (1-momentum) * extra_lr * diff_ave
                    step_cache[layer][code] = dx                
                elif update_method == 'rmsprop':
                    step_cache[layer][code] =  decay_rate * step_cache[layer][code] + (1.0 - decay_rate) * diff_ave ** 2
                    dx = -(extra_lr* diff_ave) / np.sqrt(step_cache[layer][code] + 1e-6)
                elif update_method == 'adadelta':                                                                              
                    step_cache[layer][code] = step_cache[layer][code] * decay_rate + (1.0 - decay_rate) * diff_ave ** 2           
                    dx = - np.sqrt( (step_cache2[layer][code] + smooth_eps) / (step_cache[layer][code] + smooth_eps) ) * diff_ave
                    step_cache2[layer][code] = step_cache2[layer][code] * decay_rate + (1.0 - decay_rate) * (dx ** 2)             

                codebook[layer][code] += dx
        else:
            pass

        # Maintain the not-updated layers and update the to-update layers
        W2 = codebook[layer][maskCode[layer]]
        net.params[layer][0].data[...]=W2

    if args.timing:
        print "Update codebook time:%f"%(time.time() - start_time)

