#!/bin/bash
#no argument required
thresh_list=(0.46 0.72 1.05 1.27 1.44 1.58 1.70 1.81 1.91 2.00)
thresh_list=(2.64 3.38)
folder=L1_3
# suffix=layerwise_
suffix=alex_pruned
layers=fc678

model="$CAFFE_ROOT/3_prototxt_solver/$folder/train_val.prototxt"
output="$CAFFE_ROOT/2_results/$folder/acc_before_retrain_$layers.csv"
filename_prefix="$CAFFE_ROOT/4_model_checkpoint/1_before_retrain/$folder/$suffix"
rm -rf $output

tmp1=.`date +"%T.%3N"`.tmp
tmp2=.`date +"%T.%3N"`.tmp
for data1 in ${thresh_list[@]} 
do	
	data2=`echo "$data1+1.2" | bc `
	result1=$data1
	result2=$data2
	filename1=$filename_prefix$data1"_$layers.caffemodel"		
	filename2=$filename_prefix$data2"_$layers.caffemodel"		
	# echo $result1
	# echo $result2
	rm -rf $tmp1
	rm -rf $tmp2
	# echo $CAFFE_ROOT/build/tools/caffe test --model=$model --weights=$filename1  --iterations=1 --gpu 0
	# echo $CAFFE_ROOT/build/tools/caffe test --model=$model --weights=$filename2  --iterations=1 --gpu 1
	$CAFFE_ROOT/build/tools/caffe test --model=$model --weights=$filename1  --iterations=1000 -gpu 0 >$tmp1 2>&1  &
	$CAFFE_ROOT/build/tools/caffe test --model=$model --weights=$filename2  --iterations=1000 -gpu 1 >$tmp2 2>&1  &&

	result1=$result1", "`grep -nrsI '] accuracy_top1 = ' $tmp1 | awk '{print $7 }'`
	result2=$result2", "`grep -nrsI '] accuracy_top1 = ' $tmp2 | awk '{print $7 }'`
	
	result1=$result1", "`grep -nrsI '] accuracy_top5 = ' $tmp1 | awk '{print $7 }'`
	result2=$result2", "`grep -nrsI '] accuracy_top5 = ' $tmp2 | awk '{print $7 }'`
	echo $result1
	echo $result2
	echo $result1 >> $output
	echo $result2 >> $output
done

# echo `cat $output | sort` > $output
rm -rf $tmp1
rm -rf $tmp2