#!/usr/bin/python
import numpy as np
from matplotlib.pyplot import *
import os
import sys

caffe_root = os.environ["CAFFE_ROOT"]

if len(sys.argv) == 2:
	fileName = sys.argv[1]
	data = open(fileName, "r")
	x = []
	y1 = []
	y2 = []
	y3 = []
	for line in data:
		x.append(float(line.split(',')[0]))
		y1.append(float(line.split(',')[1]))
		y2.append(float(line.split(',')[2]))
		try:
			y3.append(float(line.split(',')[3]))
		except:
			pass
		ss = np.array([40] * len(x))


	ax1 = subplot(111)
	ax2 = ax1.twinx()
	ax1.scatter(x, y1, s=ss, color='g')
	ax2.scatter(x, y2, s=ss, color='m')
	ax2.scatter(x, y3, s=ss, color='r')

	ax1.bar(x, y1, width=0.1, color='g', label='parameter count')
# 	ax1.plot(x, y1, lw=2, color='g', label='parameter count')
	ax2.plot(x, y2, lw=2, color='m', label='top-1 accuracy')
	ax2.plot(x, y3, lw=2, color='r', label='top-5 accuracy')




	xmajorLocator = MultipleLocator(0.1)  #
	xmajorFormatter = FormatStrFormatter('%5.1f')
	xminorLocator = MultipleLocator(0.1)

	ymajorLocator = MultipleLocator(0.1)
	ymajorFormatter = FormatStrFormatter('%1.2f')
	yminorLocator = MultipleLocator(0.02)

	ax1.xaxis.set_major_locator(xmajorLocator)
	ax1.xaxis.set_major_formatter(xmajorFormatter)

	ax1.yaxis.set_major_locator(ymajorLocator)
	ax1.yaxis.set_major_formatter(ymajorFormatter)


	ax1.xaxis.set_minor_locator(xminorLocator)
	ax1.yaxis.set_minor_locator(yminorLocator)

	ax1.xaxis.grid(True, which='major')
	ax1.yaxis.grid(True, which='minor')


	xlim(0, 2.3)
	ax2.set_ylim(0.31, 0.81)
	ax1.set_ylim(0, 1)


	font = {'family' : 'normal',
        		'weight' : 'normal',
          	'size'   : 18}

	matplotlib.rc('font', **font)

	ax1.set_xlabel('Threshold');
	ax1.set_ylabel('Parameter count');
	ax2.set_ylabel('Accuracy');
	grid(b=True, which='major', color='k', linestyle='-')


	ax1.legend(loc=3)
	ax2.legend(loc=1)
# 	legend([ax1, ax2])
	title ("accuracy / param count v.s threshold")
	show()

else:
    print("please pass in file name!")
