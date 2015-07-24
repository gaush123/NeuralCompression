import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.vq_maohz as scv
import pickle
import time

def kmeans_net(net, layers, num_c = 16, initials=None, method='linear',compress=True):                 
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
            if method=='linear':
                std = np.std(W)                                             
                initial_uni = np.linspace(-4 * std, 4 * std, num_c[idx] - 1)
                codebook[layer],_= scv.kmeans(W, initial_uni, compress=compress)               
            elif method == 'random':
                codebook[layer],_= scv.kmeans(W, num_c[idx]-1, compress=compress)               
            else:
                raise Exception
                
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

    if timing:
        print "Update codebook time:%f"%(time.time() - start_time)

    return codeDict, maskCode

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(step_cache={}, step_cache2={}, initial=False)
def update_codebook_net(net, codebook, codeDict, maskCode, args, update_layers=None, snapshot=None):

    start_time = time.time()
    extra_lr=args.lr
    decay_rate = args.decay_rate 
    momentum= args.momentum
    update_method= args.update

    normalize_flag = args.normalize_flag


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
            dx = np.zeros((codeBookSize))
            for code in xrange(1,codeBookSize):
                indexes = codeDict[layer][code]
                if args.average_flag:
                    diff_ave=np.sum(diff[indexes])/len(indexes)
                else:
                    diff_ave = np.sum(diff[indexes])

                if update_method == 'sgd':
                    dx[code] = -extra_lr * diff_ave
                elif update_method == 'momentum':
                    dx[code] = momentum * step_cache[layer][code] - (1-momentum) * extra_lr * diff_ave
                    step_cache[layer][code] = dx                
                elif update_method == 'rmsprop':
                    step_cache[layer][code] =  decay_rate * step_cache[layer][code] + (1.0 - decay_rate) * diff_ave ** 2
                    dx[code] = -(extra_lr* diff_ave) / np.sqrt(step_cache[layer][code] + 1e-6)
                elif update_method == 'adadelta':                                                                              
                    step_cache[layer][code] = step_cache[layer][code] * decay_rate + (1.0 - decay_rate) * diff_ave ** 2           
                    dx[code] = - np.sqrt( (step_cache2[layer][code] + smooth_eps) / (step_cache[layer][code] + smooth_eps) ) * diff_ave
                    step_cache2[layer][code] = step_cache2[layer][code] * decay_rate + (1.0 - decay_rate) * (dx ** 2)             

            if normalize_flag:
                codebook[layer] += extra_lr * np.sqrt(np.mean(codebook[layer] ** 2)) / np.sqrt(np.mean(dx ** 2)) * dx
            else:
                codebook[layer] += dx
        else:
            pass

        # Maintain the not-updated layers and update the to-update layers
        W2 = codebook[layer][maskCode[layer]]
        net.params[layer][0].data[...]=W2

    if args.timing:
        print "Update codebook time:%f"%(time.time() - start_time)

    if snapshot is not None:
        pickle.dump(codebook, open(snapshot, 'w'))

def store_all(net, codebook, dir_t, idx=0):
    net.save(dir_t + 'caffemodel%d'%idx)
    pickle.dump(codebook, open(dir_t + 'codebook%d'%idx, 'w'))

def recover_all(net, dir_t, idx=0):
    layers = net.params.keys()
    net.copy_from(dir_t + 'caffemodel%d'%idx)
    codebook = pickle.load(open(dir_t + 'codebook%d'%idx))
    maskCode = {}
    codeDict = {}
    for layer in layers:
        W = net.params[layer][0].data

        codes, _ = scv.vq(W.flatten(), codebook[layer])

        maskCode[layer] = np.reshape(codes, W.shape)
        codeBookSize = len(codebook[layer])    
        a = maskCode[layer].flatten()
        b = xrange(len(a))

        codeDict[layer]={}
        for i in xrange(len(a)):
            codeDict[layer].setdefault(a[i], []).append(b[i])

    return codebook, maskCode, codeDict


