#!/bin/bash
./build/tools/caffe test -model ./3_prototxt_solver/lenet_300_100/train_val.prototxt -weights examples/mnist/lenet_300_100_1/lenet_300_100_iter_20000.caffemodel -iterations 100 -gpu 0


