#!/bin/bash

trace=$1
folder=$2
echo $trace $folder
grep -nrsI 'Train net output #0: accuracy_top1' $trace | awk '{print $11 }' >  $CAFFE_ROOT/2_results/$folder/train_acc_top1.csv
grep -nrsI 'Train net output #1: accuracy_top5' $trace | awk '{print $11 }' >  $CAFFE_ROOT/2_results/$folder/train_acc_top5.csv
grep -nrsI 'Train net output #2: loss' $trace | awk '{print $11 }' >  $CAFFE_ROOT/2_results/$folder/train_loss.csv
   
grep -nrsI 'Test net output #0: accuracy_top1' $trace | awk '{print $11 }' >  $CAFFE_ROOT/2_results/$folder/test_acc_top1.csv
grep -nrsI 'Test net output #1: accuracy_top5' $trace | awk '{print $11 }' >  $CAFFE_ROOT/2_results/$folder/test_acc_top5.csv
grep -nrsI 'Test net output #2: loss' $trace | awk '{print $11 }' >  $CAFFE_ROOT/2_results/$folder/test_loss.csv
#echo "top_1========="
#grep -nrsI 'Test net output #0: accuracy_top1' $trace | awk '{print $11 }' 
#echo "top_5========="
#grep -nrsI 'Test net output #1: accuracy_top5' $trace | awk '{print $11 }' 
#echo "loss=========="
#grep -nrsI 'Test net output #2: loss' $trace | awk '{print $11 }'
