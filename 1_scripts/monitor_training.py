#!/usr/bin/python
import numpy as np
from matplotlib.pyplot import *
import os
import sys


def plot_figure(fileName):
	data = open(fileName,"r")
	y=[]
	for line in data:
		y.append(float(line.split()[0]))
	
	x=range(len(y))
	figure
	plot(x, y)

def plot_ave_figure(fileName):
	data = open(fileName,"r")
	y=[]
	line_cnt=0
	for line in data:
		tmp_y=[]
		tmp_y.append(float(line.split()[0]))		
		line_cnt=line_cnt+1
		if line_cnt == 10:
			y.append(reduce(lambda x, y: x + y, tmp_y) / float(len(tmp_y)))
			line_cnt=0
	x=range(len(y))
	figure
	plot(x, y)

caffe_root = os.environ["CAFFE_ROOT"]
if len(sys.argv)==2: 
	fileName = sys.argv[1]
	os.system(caffe_root+'/1_scripts/extract_trace.sh '+fileName)
	fileName1 = caffe_root+"/2_results/train_loss.csv"
	fileName2 = caffe_root+"/2_results/train_accuracy_top1.csv"
	fileName3 = caffe_root+"/2_results/train_accuracy_top5.csv"

	fileName4 = caffe_root+"/2_results/test_loss.csv"
	fileName5 = caffe_root+"/2_results/test_accuracy_top1.csv"
	fileName6 = caffe_root+"/2_results/test_accuracy_top5.csv"

	plot_ave_figure(fileName1)
	xlabel("iteration")
	ylabel("training loss")
	title ("training loss")
	show()
	plot_ave_figure(fileName2)
	xlabel("iteration")
	ylabel("training accuracy_top1")
	title ("training accuracy_top1")
	show()
	plot_ave_figure(fileName3)
	xlabel("iteration")
	ylabel("training accuracy_top5")
	title ("training accuracy_top5")
	show()
	
	plot_figure(fileName4)
	xlabel("iteration")
	ylabel("test loss")
	title ("test loss")
        show()
	plot_figure(fileName5)
	xlabel("iteration")
	ylabel("test accuracy_top1")
	title ("test accuracy_top1")
        show()        
	plot_figure(fileName6)
	xlabel("iteration")
	ylabel("test accuracy_top5")
	title ("test accuracy_top5")
	show()
else:
        print("please pass in file name!")
