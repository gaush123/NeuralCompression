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
caffe.set_device(3)
prototxt = '3_prototxt_solver/L2/train_val.prototxt'          
solver = '3_prototxt_solver/L2/finetune_solver.prototxt'      
caffemodel = '4_model_checkpoint/alexnet/alexnet9x.caffemodel'

solver = caffe.SGDSolver(solver) 
solver.net.copy_from(caffemodel) 
net = solver.net                 

'''
layers = filter(lambda x:'conv' in x or 'fc' in x, net.params.keys())
fout = open(xls_file,'w')
analyze_param(net, layers, fout)
fout.close()

command = caffe_root + "/build/tools/caffe test --model=" + prototxt + " --weights=" + caffemodel + " --iterations=1000 --gpu 2 "
print command
os.system(command)

'''
