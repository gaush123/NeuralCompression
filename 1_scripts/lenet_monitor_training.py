#!/usr/bin/python
import numpy as np
from matplotlib.pyplot import *
import os
import sys

folder = "//"
def analyze_log(fileName):
	data = open(fileName, "r")
	y = []
	for line in data:
		y.append(float(line.split()[0]))

	return y

def analyze_log_ave(fileName):
	data = open(fileName, "r")
	y = []
	line_cnt = 0
	for line in data:
		tmp_y = []
		tmp_y.append(float(line.split()[0]))
		line_cnt = line_cnt + 1
		if line_cnt == 10:
			y.append(reduce(lambda x, y: x + y, tmp_y) / float(len(tmp_y)))
			line_cnt = 0
	return y

caffe_root = os.environ["CAFFE_ROOT"]
if len(sys.argv) == 2:
	fileName = sys.argv[1]
	os.system(caffe_root + '/1_scripts/' + folder + 'lenet_extract_trace.sh ' + fileName)
	fileName1 = caffe_root + "/2_results/" + folder + "train_loss.csv"
	fileName2 = caffe_root + "/2_results/" + folder + "train_acc_top1.csv"
	fileName4 = caffe_root + "/2_results/" + folder + "test_loss.csv"
	fileName5 = caffe_root + "/2_results/" + folder + "test_acc_top1.csv"

	y_loss = analyze_log_ave(fileName1)
	y_top1 = analyze_log_ave(fileName2)


	x = [x * 100 for x in xrange(len(y_loss))]
	subplot(1, 3, 1)
	plot(x, y_loss)
	title ("training loss")

	subplot(1, 3, 2)
	plot(x, y_top1, 'r')
	title ("training accuracy_top1")

	draw()

	figure()

	y_loss = analyze_log(fileName4)
	y_top1 = analyze_log(fileName5)
	top1_ori, top5_ori = (0.9852 * 100, 0.9908 * 100)

	print "==============result============="
	print "original accuracy  (%.2f%%, %.2f%%)" % (top1_ori, top5_ori)

	print y_top1
# 	print y_loss
	top1 = np.max(y_top1) * 100
	idx = np.argmax(y_top1)
	print "\nbest top1 accuracy (*%.2f%%*)" % (top1)
	print "=> accuracy loss   (%.2f%% )" % (top1_ori - top1)
	print "best loss = %.2f" % (np.min(y_loss))
# 	sys.exit(0)
	x = [x * 5000 for x in xrange(len(y_loss))]
	subplot(1, 3, 1)
	plot(x, y_loss)
	title ("test loss")

	subplot(1, 3, 2)
	plot(x, y_top1, 'r')
	title ("test accuracy_top1")


	show()

else:
    print("please pass in file name!")
