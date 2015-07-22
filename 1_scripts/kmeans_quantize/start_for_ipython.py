'''
Created on Apr 18, 2015

@author: huizi, songhan
'''

import sys
import os
import numpy as np
import matplotlib.pyplot as plt


# os.system("cd $CAFFE_ROOT")
caffe_root = os.environ["CAFFE_ROOT"]
os.chdir(caffe_root)
print caffe_root
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()
caffe.set_device(0)
prototxt = '/home/maohz12/pruning/3_prototxt_solver/vgg16/VGG_ILSVRC_16_layers_deploy.prototxt'
caffemodel = '/home/maohz12/pruning/4_model_checkpoint/vgg16/vgg16_12x.caffemodel'
xls_file = '2_results/VGG16/res.xls'

net = caffe.Classifier(prototxt, caffemodel, caffe.TEST)

'''
layers = filter(lambda x:'conv' in x or 'fc' in x, net.params.keys())
fout = open(xls_file,'w')
analyze_param(net, layers, fout)
fout.close()

command = caffe_root + "/build/tools/caffe test --model=" + prototxt + " --weights=" + caffemodel + " --iterations=1000 --gpu 2 "
print command
os.system(command)

'''
