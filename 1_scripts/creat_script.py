'''
Created on Apr 19, 2015

@author: songhan
'''
# -*- coding: utf-8 -*-
folder = "L2_conv"
# thresh_list = ["0.50", "0.80", "1.05", "1.20", "1.27", "1.44", "1.58", "1.70", "1.81", "1.91", "2.00"]
thresh_list = ["0.08", "0.25", "0.69", "1.06", "1.35", "1.59", "1.80", "1.99", "2.16", "2.32", "2.76", "3.51"]
thresh_list = ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
for i, thresh in enumerate(thresh_list):
    num = str(i % 4)


#     command1 = "./build/tools/caffe train -solver ./3_prototxt_solver/" + folder + "/solver" + thresh + ".prototxt -weights ./4_model_checkpoint/1_before_retrain/" + folder + "/alex_pruned_" + thresh + "_678half.caffemodel -gpu " + num + " >> 2_results/" + thresh + ".txt 2>&1 &"
    command1 = "./build/tools/caffe train -solver ./3_prototxt_solver/" + folder + "/solver1.44_" + thresh + ".prototxt -weights ./4_model_checkpoint/1_before_retrain/" + folder + "/conv_all_" + thresh + "_all.caffemodel -gpu " + num + " >> 2_results/conv" + thresh + ".txt 2>&1 &"
    print command1
    print "\n"

#     command2 = "./build/tools/caffe train -solver ./3_prototxt_solver/" + folder + "/solver" + thresh + "_small.prototxt -weights ./4_model_checkpoint/1_before_retrain/" + folder + "/alex_pruned_" + thresh + "_678half.caffemodel -gpu " + num + " >> 2_results/" + thresh + "_small.txt 2>&1 &"
# 	print command2
