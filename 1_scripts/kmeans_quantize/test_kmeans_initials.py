
from kmeans import *

def get_initials(data, bits, types_init=3):
    std = np.std(data)
    iters = types_init / 3
    res = []
    initial1 = 'random'
    min_W = np.min(data)
    max_W = np.max(data)
    initial2 = np.linspace(min_W, max_W, 2 ** bits -1)

    thresh = np.linspace(-5 * std, 5 * std, 2 ** (bits + 2))
    number_down_thresh = np.array(map(lambda x: np.count_nonzero(data < x), thresh))

    # Optional: whether to soften the density curve?
    wanted_division = np.linspace(0, data.size, 2 ** (bits + 1)-2)[1::2]
    print "wanted divison", wanted_division
    thresh = np.append(max_W, thresh)
    initial3 = np.array(map(lambda x: thresh[np.count_nonzero(number_down_thresh < x)], wanted_division))
    epsilon = 0.05 * std / (2 ** bits)
    initial3 += (np.array(range(len(initial3))) - len(initial3) / 2) * epsilon

    return [initial1] * (types_init - 2) + [initial2, initial3]


def eval_initials(prototxt, caffemodel, bits_list, log, layer):
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    types_init = 3

    accu_top1 = np.zeros((types_init, len(bits_list)))
    accu_top5 = np.zeros((types_init, len(bits_list)))
    data = net.params[layer][0].data.copy()
    for j, bits in enumerate(bits_list):
        for i, initial_setting in enumerate(get_initials(data, bits, types_init)):
            num_c = 2 ** bits
            net = caffe.Net(prototxt, caffemodel, caffe.TEST)
            print "====================Notes================="
            print "Current proceedings:", i, "bits:", bits
            print "===============Accuracy================="
            print accu_top1, accu_top5

            codebook = kmeans_net(net, [layer], num_c, initial_setting)
            quantize_net(net, codebook)

            net.save(caffemodel + '.quantize')
            log_new = log + "_" + "_%dbits" % bits + "initial_setting%d" % i
            command = caffe_root + "/build/tools/caffe test --model=" + prototxt + " --weights=" + caffemodel + ".quantize --iterations=%d --gpu 2 2>" % iters + log_new
            print command
            os.system(command)
            os.system('tail -n 3 ' + log_new)

            accu1, accu2 = parse_caffe_log(log_new)
            accu_top1[i, j] = accu1
            accu_top5[i, j] = accu2

    return (accu_top1, accu_top5)

def main_init():
    bits_list = [1, 2, 3, 4, 5, 6, 7, 8]
    layer = 'conv2'
    caffemodel = '4_model_checkpoint/alexnet/alexnet9x.caffemodel'
    accu_top1, accu_top5 = eval_initials(prototxt, caffemodel, bits_list, log, layer)

    np.save(dir_t + 'accu_top1_initials_' + layer, accu_top1)
    np.save(dir_t + 'accu_top5_initials_' + layer, accu_top5)


def test():
    data = np.random.normal(0, 1, 1000)
    initis = get_initials(data, 2, 2)
    print initis[1]
    initis = get_initials(data, 1, 2)
    print initis[1]
    initis = get_initials(data, 4, 2)
    print initis[1]

if __name__ == "__main__":
    main_init()
    # test()
