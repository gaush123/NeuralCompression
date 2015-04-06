#!/bin/bash

trace=$1
echo $trace
grep -nrsI 'Train net output #0: accuracy_top1' $trace | awk '{print $11 }' >  $CAFFE_ROOT/2_results/train_accuracy_top1.csv
grep -nrsI 'Train net output #1: accuracy_top5' $trace | awk '{print $11 }' >  $CAFFE_ROOT/2_results/train_accuracy_top5.csv
grep -nrsI 'Train net output #2: loss' $trace | awk '{print $11 }' >  $CAFFE_ROOT/2_results/train_loss.csv
   
echo "here"
grep -nrsI 'Test net output #0: accuracy_top1' $trace | awk '{print $11 }' >  $CAFFE_ROOT/2_results/test_accuracy_top1.csv
grep -nrsI 'Test net output #1: accuracy_top5' $trace | awk '{print $11 }' >  $CAFFE_ROOT/2_results/test_accuracy_top5.csv
grep -nrsI 'Test net output #2: loss' $trace | awk '{print $11 }' >  $CAFFE_ROOT/2_results/test_loss.csv

