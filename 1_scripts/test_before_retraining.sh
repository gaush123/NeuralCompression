#!/bin/bash
#no argument required
# thresh_list=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3)
thresh_list=(0.08 2.76 3.51)
folder=L1_3

model="$CAFFE_ROOT/3_prototxt_solver/$folder/train_val.prototxt"
output="$CAFFE_ROOT/2_results/$folder/acc_before_retrain_678half.csv"
filename_prefix="$CAFFE_ROOT/4_model_checkpoint/1_before_retrain/$folder/alex_pruned_"
rm -rf $output

tmp=`date +"%T.%3N"`.tmp
for data in ${thresh_list[@]} 
do	
	filename=$filename_prefix$data"_678half.caffemodel"		
	result=$data	
	rm -rf $tmp
	echo $CAFFE_ROOT/build/tools/caffe test --model=$model --weights=$filename  --iterations=1000 --gpu 3
	$CAFFE_ROOT/build/tools/caffe test --model=$model --weights=$filename  --iterations=1000 --gpu 3 >$tmp 2>&1  &&
	result=$result", "`grep -nrsI '] accuracy_top1 = ' $tmp | awk '{print $7 }'`
	result=$result", "`grep -nrsI '] accuracy_top5 = ' $tmp | awk '{print $7 }'`
	echo $result
	echo $result >> $output
done


rm -rf $tmp
