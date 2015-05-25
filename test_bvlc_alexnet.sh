#!/bin/bash
./build/tools/caffe test --model=./models/bvlc_alexnet/train_val.prototxt --weights=models/bvlc_alexnet/bvlc_alexnet.caffemodel  --iterations=1000 --gpu 0
#--weights=models/bvlc_alexnet/alex_svd512.caffemodel  --iterations=1000 --gpu 1



