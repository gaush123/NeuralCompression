#!/bin/bash
./build/tools/caffe test -model ./3_prototxt_solver/lenet5/train_val.prototxt -weights examples/mnist/lenet_tmp_iter_10000.caffemodel -iterations 100 -gpu 0


