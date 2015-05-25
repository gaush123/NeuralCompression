'''
Created on Apr 19, 2015

@author: songhan
'''
# -*- coding: utf-8 -*-
folder = "L2_conv"
folder = "lenet5"
# thresh_list = ["0.50", "0.80", "1.05", "1.20", "1.27", "1.44", "1.58", "1.70", "1.81", "1.91", "2.00"]
# threshold_list = ["0.08", "0.25", "0.69", "1.06", "1.35", "1.59", "1.80", "1.99", "2.16", "2.32", "2.76", "3.51"]
threshold_list = [  1.46, 1.49, 1.50, 1.51, 1.53, 1.55, 1.56, 1.61]
for i, thresh in enumerate(threshold_list):
    num = str(i % 4)
    thresh = str(thresh)

#     command1 = "./build/tools/caffe train -solver ./3_prototxt_solver/" + folder + "/solver" + thresh + ".prototxt -weights ./4_model_checkpoint/1_before_retrain/" + folder + "/alex_pruned_" + thresh + "_678half.caffemodel -gpu " + num + " >> 2_results/" + thresh + ".txt 2>&1 &"
#     command1 = "./build/tools/caffe train -solver ./3_prototxt_solver/" + folder + "/solver1.44_" + thresh + ".prototxt -weights ./4_model_checkpoint/1_before_retrain/" + folder + "/conv_all_" + thresh + "_all.caffemodel -gpu " + num + " >> 2_results/conv" + thresh + ".txt 2>&1 &"
#     command1 = "./build/tools/caffe train -solver ./3_prototxt_solver/" + folder + "/lenet_solver_retrain" + thresh + ".prototxt -weights ./4_model_checkpoint/1_before_retrain/" + folder + "/lenet_" + thresh + "_all.caffemodel -gpu " + num + " >> 2_results/lenet5" + thresh + ".txt 2>&1 &"
#     command1 = "./build/tools/caffe train -solver ./3_prototxt_solver/" + folder + "/lenet_solver_retrain.prototxt -weights ./4_model_checkpoint/1_before_retrain/" + folder + "/lenet_" + thresh + "_all.caffemodel -gpu " + num + " > 2_results/lenet5" + thresh + ".txt 2>&1 &"
    command1 = "./build/tools/caffe train -solver ./3_prototxt_solver/" + folder + "/lenet_multistep_solver.prototxt -weights ./4_model_checkpoint/1_before_retrain/" + folder + "/lenet_" + thresh + "_relu_all.caffemodel -gpu " + num + " > 2_results/lenet5" + thresh + ".txt 2>&1 &"
    print command1
    print "\n"

#     command2 = "./build/tools/caffe train -solver ./3_prototxt_solver/" + folder + "/solver" + thresh + "_small.prototxt -weights ./4_model_checkpoint/1_before_retrain/" + folder + "/alex_pruned_" + thresh + "_678half.caffemodel -gpu " + num + " >> 2_results/" + thresh + "_small.txt 2>&1 &"
# 	print command2
