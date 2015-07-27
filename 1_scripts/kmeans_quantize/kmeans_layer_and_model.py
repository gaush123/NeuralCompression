'''
Created on Apr 18, 2015

@author: huizi
'''


from kmeans_by_layer import *

caffe.set_device(0)
def main2():
    bits_list = [2, 3, 4, 6, 8]

    dir_t = '4_model_checkpoint/alexnet/'
    models = [dir_t + 'alexnet.caffemodel',
        dir_t + 'alexnet6x.caffemodel',
        dir_t + 'alexnet9x.caffemodel']

    accu_top1 = []
    accu_top5 = []
    for model in models:
        accu_top1_temp, accu_top5_temp = eval_accu_layerwise(prototxt, model, bits_list, log)
        accu_top1.append(accu_top1_temp)
        accu_top5.append(accu_top5_temp)

    accu_top1 = np.array(accu_top1)
    accu_top5 = np.array(accu_top5)

    np.save(dir_t + 'accu_top1_3models', accu_top1)
    np.save(dir_t + 'accu_top5_3models', accu_top5)

if __name__ == "__main__":
    main2()

