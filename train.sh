#!/bin/bash
./build/tools/caffe train -solver=./models/bvlc_reference_caffenet/solver.prototxt --gpu $1



