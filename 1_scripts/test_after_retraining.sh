#!/bin/bash

# folder=L2
# suffix=layerwise_
# layers=fc6

# model="$CAFFE_ROOT/3_prototxt_solver/$folder/train_val.prototxt"
# output="$CAFFE_ROOT/2_results/$folder/acc_before_retrain_$layers.csv"
# filename_prefix="$CAFFE_ROOT/4_model_checkpoint/1_before_retrain/$folder/$suffix"

model="$CAFFE_ROOT/3_prototxt_solver/train_val_prune1.prototxt"
# filename_prefix="$CAFFE_ROOT/4_model_checkpoint/2_after_retrain/prune$1_iter_"
filename_prefix="/cnn/caffemodel/caffe_alexnet_train1_iter_"

if [ "$1" -gt "$2" ]; then
	let start=$2
	let end=$1
else 
	let start=$1
	let end=$2
fi
echo start=$start, end=$end
output=result_$start_$end.csv
tmp=`date +"%T.%3N"`.tmp
cd $CAFFE_ROOT

for iter in $(seq $end -5000 $start);  do  
    filename=$filename_prefix$iter.caffemodel
	result=$iter
	rm -rf $tmp
	# echo  $filename
	# echo "./build/tools/caffe test --model=$model  --weights=$filename  --iterations=1000 --gpu 1"
	$CAFFE_ROOT/build/tools/caffe test --model=$model  --weights=$filename  --iterations=1000 --gpu 0  >>$tmp 2>&1 &&
	result=$result", "`grep -nrsI '] accuracy_top1 = ' $tmp | awk '{print $7 }'` 
	result=$result", "`grep -nrsI '] accuracy_top5 = ' $tmp | awk '{print $7 }'`
	result=$result", "`grep -nrsI '] loss = ' $tmp | awk '{print $7 }'`
	
	echo $result
	echo $result >> $output
done


rm -rf $tmp
