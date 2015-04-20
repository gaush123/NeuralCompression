'''
Created on Apr 19, 2015

@author: songhan
'''
# -*- coding: utf-8 -*-
folder = "L2"
thresh_list = ["0.50", "0.80", "1.05", "1.20", "1.27", "1.44", "1.58", "1.70", "1.81", "1.91", "2.00"]
for i, thresh in enumerate(thresh_list):
    num = str(i % 4)


    command1 = "./build/tools/caffe train -solver ./3_prototxt_solver/" + folder + "/solver" + thresh + ".prototxt -weights ./4_model_checkpoint/1_before_retrain/" + folder + "/alex_pruned_" + thresh + "_678half.caffemodel -gpu " + num + " >> 2_results/" + thresh + ".txt 2>&1 &"
    print command1

    command2 = "./build/tools/caffe train -solver ./3_prototxt_solver/" + folder + "/solver" + thresh + "_small.prototxt -weights ./4_model_checkpoint/1_before_retrain/" + folder + "/alex_pruned_" + thresh + "_678half.caffemodel -gpu " + num + " >> 2_results/" + thresh + "_small.txt 2>&1 &"
	print command2
