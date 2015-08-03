from kmeans import *

sys.path.insert(0, caffe_root + '1_scripts/finetune')

from utils import store_all, recover_all

def kmeans_random_disturb(net, layers, num_c = 16, initials=None):
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
            disturb = np.random.normal(loc=0.0, scale = std/num_c[idx] * 0.1, size = initial_uni.shape)
            initial_uni += disturb
            codebook[layer],_= scv.kmeans(W, initial_uni, compress=False)

        elif type(initials) == type(np.array([])):
            codebook[layer],_= scv.kmeans(W, initials)
        else:
            print type(initials)
            return None
        codebook[layer] = np.append(0.0, codebook[layer])
        print "codebook size:", len(codebook[layer])
    return codebook

net = caffe.Net(prototxt, caffemodel, caffe.TEST)
# layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())
layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())

choice = [64,16]
num_c = [choice[0]]  * (len(layers)-3) + [choice[1]] * 3

best_1 = 0.0
best_5 = 0.0
dir_t_res = '2_results/kmeans/alexnet/best_6_4/'
for iter in range(200):

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    codebook = kmeans_net(net, layers, num_c)
    quantize_net(net, codebook)
    net.save(caffemodel + '.quantize')
    command = caffe_root + "/build/tools/caffe test --model=" + prototxt + " --weights=" + caffemodel + ".quantize --iterations=%d --gpu 2 2>"%iters +log + "new"
    os.system(command)
    top1, top5 = parse_caffe_log(log  +"new")
    if top1 + top5 > best_1 + best_5:
        best_1 = top1
        best_5 = top5
        store_all(net, codebook, dir_t_res)

    print "Iters: ", iter
    print top1, top5
    print best_1, best_5
