#!/usr/bin/python
import numpy as np
from matplotlib.pyplot import *
import os
import sys

caffe_root = os.environ["CAFFE_ROOT"]

def read_file(fileName):
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
    return (x, y1, y2, y3)


if len(sys.argv) == 3:
    fileName = sys.argv[1]
    (x, y1, y2, y3) = read_file(fileName)
    ss = np.array([40] * len(x))
    ax1 = subplot(111)
    ax1.plot(y1, y3, lw=2, color='m', label=fileName)

    fileName = sys.argv[2]
    (x, y1, y2, y3) = read_file(fileName)
    ax1.plot(y1, y3, lw=2, color='g', label=fileName)



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


    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.5, 0.81)


    font = {'family' : 'normal',
                'weight' : 'normal',
              'size'   : 18}

    matplotlib.rc('font', **font)

    ax1.set_xlabel('parameter count');
    ax1.set_ylabel('top-5 accuracy');
    grid(b=True, which='major', color='k', linestyle='-')


    ax1.legend(loc=1)
    title ("accuracy v.s param count ")
    show()

else:
    print("please pass in 2 file name!")

