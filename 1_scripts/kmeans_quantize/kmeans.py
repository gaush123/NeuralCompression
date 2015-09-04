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
caffe.set_device(0)
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

def kmeans_net(net, layers, num_c=16, initials=None, snapshot=False, alpha=0.0):
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
        if initials is None:  # Default: uniform sample
            min_W = np.min(W)
            max_W = np.max(W)
            initial_uni = np.linspace(min_W, max_W, num_c[idx] - 1)
            ''' # Legacy
            std = np.std(W)
            initial_uni = np.linspace(std * -4, std * 4, num_c[idx] - 1)
            '''

            if snapshot and layer=='fc6':
                codebook[layer], _, codebook_history = scv.kmeans(W, initial_uni, compress=False, snapshot=True, alpha=alpha)
            else:
                codebook[layer], _= scv.kmeans(W, initial_uni, compress=False, alpha=alpha)

            '''
            codebook[layer],_= scv.kmeans(W, num_c[idx] - 1)
            '''
        elif type(initials) == type(np.array([])):
            codebook[layer], _ = scv.kmeans(W, initials)
        elif initials == 'random':
            codebook[layer], _ = scv.kmeans(W, num_c[idx]-1)
            
        codebook[layer] = np.append(0.0, codebook[layer])
        print "codebook size:", len(codebook[layer])

    if snapshot:
        return codebook, codebook_history

    return codebook
def stochasitc_quantize2(W, codebook):

    mask = W[:, np.newaxis] - codebook

    mask_neg = mask
    mask_neg[mask_neg > 0.0] -= 99999.0
    max_neg = np.max(mask_neg, axis=1)
    max_code = np.argmax(mask_neg, axis=1)

    mask_pos = mask
    mask_pos += 99999.0
    min_code = np.argmin(mask_pos, axis=1)
    min_pos = np.min(mask_pos, axis=1)

    rd = np.random.uniform(low=0.0, high=1.0, size=(len(W)))
    thresh = min_pos.astype(np.float32) / (min_pos - max_neg)

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
    try:
        res = map(lambda x: float(x.split()[-1]), lines[-3:-1])
    except Exception as e:
        print e
        res = [0.0,0.0]
    return res

def test_quantize_accu(choice = [6,4], layers = None, alpha=1.0):
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    if layers is None:
        layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())

    if len(choice) == 2:                                                  
        if option == 'lenet5':                                            
            bits_list = [choice[0]] * (len(layers) - 2) + [choice[1]] * 2 
        else:                                                             
            bits_list = [choice[0]] * (len(layers) - 3) + [choice[1]] * 3 
    else:                                                                 
        assert len(choice) == len(layers)                                 
        bits_list = choice                                                

    num_c = map(lambda x: 2 ** x, bits_list )
    codebook = kmeans_net(net, layers, num_c, alpha=alpha)
    quantize_net(net, codebook)
    net.save(caffemodel + '.quantize')
    command = caffe_root + "/build/tools/caffe test --model=" + prototxt + " --weights=" + caffemodel + ".quantize --iterations=%d --gpu 0 2>"%iters +log + "new"
    os.system(command)
    accu_top1, accu_top5 = parse_caffe_log(log + 'new')
    return (accu_top1, accu_top5)



def main(choice=[64, 16], snapshot=False):
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
# layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())
    layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())

# Evaluate the origina accuracy
    command = caffe_root + "/build/tools/caffe test --model=" + prototxt + " --weights=" + caffemodel + " --iterations=%d --gpu 1 2>" % iters + log
    # print command
    # os.system(command)
    # os.system('tail -n 3 '+ log)

    num_c = [choice[0]] * (len(layers) - 3) + [choice[1]] * 3
    if snapshot:
        codebook, codebook_history = kmeans_net(net, layers, num_c, snapshot=True)
        pickle.dump(codebook_history, open(dir_t + 'codebook_history_4std.pkl', 'w'))
    else:
        codebook = kmeans_net(net, layers, num_c, snapshot=False)

    pickle.dump(codebook, open(dir_t + 'codebook.pkl', 'w'))
    quantize_net(net, codebook)

# Evaluate the new model's accuracy
    net.save(caffemodel + '.quantize')
    command = caffe_root + "/build/tools/caffe test --model=" + prototxt + " --weights=" + caffemodel + ".quantize --iterations=%d --gpu 1 2>"%iters +log + "new"
    #print command
    print choice
    os.system(command)
    os.system('tail -n 3 '+ log + 'new')

    top_1,top_5 = parse_caffe_log(log + 'new')
    with open('results_%s'%option,'a+') as f:
        f.write('%d %d \n%f\n%f\n'%(choice[0], choice[1], top_1, top_5))

def main2():
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
# layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())
    layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())

# Evaluate the origina accuracy
    command = caffe_root + "/build/tools/caffe test --model=" + prototxt + " --weights=" + caffemodel + " --iterations=%d --gpu 1 2>" % iters + log
    print command
    # os.system(command)
    os.system('tail -n 3 ' + log)

    bits = [8, 8, 7, 6, 6, 3, 3, 4]
    num_c = 2 ** bits
    codebook = kmeans_net(net, layers, num_c)

    pickle.dump(codebook, open(dir_t + 'codebook.pkl', 'w'))

    quantize_net(net, codebook)

# Evaluate the new model's accuracy
    net.save(caffemodel + '.quantize')
    command = caffe_root + "/build/tools/caffe test --model=" + prototxt + " --weights=" + caffemodel + ".quantize --iterations=%d --gpu 1 2>" % iters + log + "new"
    print command
    os.system(command)
    os.system('tail -n 3 ' + log + 'new')

def test_weighted_kmeans():

    settings = [[4,2],[5,2], [8,3], [10,5]]
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    f = open('tmp/alphatest.csv','w')
    for setting in settings:
        for alpha in alphas:
            f.write('%d, %d, %f, %f, %f\n'%((tuple(setting) + (alpha,) + test_quantize_accu(setting, alpha=alpha))))
    f.close()

if __name__ == "__main__":
    '''
    main([256,8])
    main([64,8])
    main([64,16])
    main2()
    '''
    main([256,16])
    # main([64,16])
    # print test_quantize_accu([3], ['conv5'], alpha=1.0)
    # test_weighted_kmeans()
