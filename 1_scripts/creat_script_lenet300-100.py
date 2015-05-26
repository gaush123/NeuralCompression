'''
Created on Apr 19, 2015

@author: songhan
'''

folder = "lenet_300_100"
prefix = "/lenet300_100_"
suffix = "_all"
threshold_list = [  1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
for i, thresh in enumerate(threshold_list):
    num = str(i % 4)
    thresh = str(thresh)

    command1 = "./build/tools/caffe train -solver ./3_prototxt_solver/" + folder + "/solver_retrain.prototxt -weights ./4_model_checkpoint/1_before_retrain/" + folder + prefix + thresh + suffix + ".caffemodel -gpu " + num + " > 2_results/" + folder + thresh + ".txt 2>&1 &"
    print command1
    print "\n"

#     command2 = "./build/tools/caffe train -solver ./3_prototxt_solver/" + folder + "/solver" + thresh + "_small.prototxt -weights ./4_model_checkpoint/1_before_retrain/" + folder + "/alex_pruned_" + thresh + "_678half.caffemodel -gpu " + num + " >> 2_results/" + thresh + "_small.txt 2>&1 &"
# 	print command2
