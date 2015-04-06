#!/bin/bash
#no argument required
thresh_list=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2,2.3)

model="$CAFFE_ROOT/models/bvlc_alexnet/train_val.prototxt"
output="$CAFFE_ROOT/2_results/accuracy_before_retrain_678half.csv"
filename_prefix="$CAFFE_ROOT/4_model_checkpoint/1_before_retrain/alex_pruned_"
rm -rf $output

for data in ${thresh_list[@]} 
do	
	echo $data
	filename=$filename_prefix$data"_678half.caffemodel"	
	echo  $filename
	result=$data	
	rm -rf lala
	echo $CAFFE_ROOT/build/tools/caffe test --model=$model --weights=$filename  --iterations=1000 --gpu 0
	$CAFFE_ROOT/build/tools/caffe test --model=$model --weights=$filename  --iterations=1000 --gpu 0 >lala 2>&1  &&
	result=$result", "`grep -nrsI '] accuracy_top1 = ' lala | awk '{print $7 }'`
	result=$result", "`grep -nrsI '] accuracy_top5 = ' lala | awk '{print $7 }'`
	echo $result >> $output
done
