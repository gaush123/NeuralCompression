#!/usr/bin/python
import numpy as np
from matplotlib.pyplot import *
import os
import sys

caffe_root = os.environ["CAFFE_ROOT"]

if len(sys.argv)==2: 
	fileName = sys.argv[1]
	data = open(fileName,"r")
	x=[]
	y1=[]
	y5=[]
	for line in data:		
		x.append(float(line.split(',')[0]))
		y1.append(float(line.split(',')[1]))
		y5.append(float(line.split(',')[2]))
		ss=np.array([20]*len(x))

	ax = subplot(111)
	scatter(x, y1,  s=ss)
	scatter(x, y5,  s=ss)	
	
	title ("test accuracy")	

	xmajorLocator   = MultipleLocator(0.1) #
	xmajorFormatter = FormatStrFormatter('%5.1f') 
	xminorLocator   = MultipleLocator(0.01) 
	  	 
	ymajorLocator   = MultipleLocator(0.05) 
	ymajorFormatter = FormatStrFormatter('%1.4f') 
	yminorLocator   = MultipleLocator(0.01) 
	  
	   
	  	   	  
	
	ax.xaxis.set_major_locator(xmajorLocator)  
	ax.xaxis.set_major_formatter(xmajorFormatter)  
	  
	ax.yaxis.set_major_locator(ymajorLocator)  
	ax.yaxis.set_major_formatter(ymajorFormatter)  
	  
	
	ax.xaxis.set_minor_locator(xminorLocator)  
	ax.yaxis.set_minor_locator(yminorLocator)  
	  
	ax.xaxis.grid(True, which='major') 
	ax.yaxis.grid(True, which='minor') 


	xlim(0, 2.3)

	show()
	
else:
    print("please pass in file name!")
