#!/bin/bash

model="$CAFFE_ROOT/3_prototxt_solver/train_val_prune1.prototxt"
filename_prefix="$CAFFE_ROOT/4_model_checkpoint/2_after_retrain/prune$1_iter_"

if [ "$2" -gt "$3" ]; then
	let start=$3
	let end=$2
else 
	let start=$2
	let end=$3
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
