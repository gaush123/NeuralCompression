#!/bin/bash
# no argument required
# thresh_list=(0.46 0.72 1.05 1.27 1.44 1.58 1.70 1.81 1.91 2.00)
# thresh_list=(0 0.25 0.69 1.06 1.35 1.59 1.80 1.99 2.16 2.32)
thresh_list=(0.08 2.32 2.64 3.38)
folder=L1_3
#suffix="fc678"
suffix="678half"
#suffix_2="layerwise_"
suffix_2="alex_pruned_"

model="$CAFFE_ROOT/3_prototxt_solver/$folder/train_val.prototxt"
output="$CAFFE_ROOT/2_results/$folder/acc_before_retrain_$suffix.csv"
filename_prefix="$CAFFE_ROOT/4_model_checkpoint/1_before_retrain/$folder/$suffix_2"
rm -rf $output

tmp1=.`date +"%T.%3N"`.tmp
tmp2=.`date +"%T.%3N"`.tmp
tmp3=.`date +"%T.%3N"`.tmp
tmp4=.`date +"%T.%3N"`.tmp



# for data1 in ${thresh_list[@]}
size=${#thresh_list[@]}
echo $size 
end=`echo "$size/4" | bc `
for i in $(seq 0 $end);
do	
	echo $i
    data1=${thresh_list[4*i]}
    data2=${thresh_list[4*i+1]}
    data3=${thresh_list[4*i+2]}
    data4=${thresh_list[4*i+3]}
	
	data1=`printf "%.2f" $data1`
	data2=`printf "%.2f" $data2`
	data3=`printf "%.2f" $data3`
	data4=`printf "%.2f" $data4`

	result1=$data1
	result2=$data2
	result3=$data3
	result4=$data4
	filename1=$filename_prefix$data1"_$suffix.caffemodel"		
	filename2=$filename_prefix$data2"_$suffix.caffemodel"		
	filename3=$filename_prefix$data3"_$suffix.caffemodel"		
	filename4=$filename_prefix$data4"_$suffix.caffemodel"		
	echo $result1
	echo $result2
	echo $result3
	echo $result4
	rm -rf $tmp1
	rm -rf $tmp2
	rm -rf $tmp3
	rm -rf $tmp4
# 	echo ./build/tools/caffe test --model=$model --weights=$filename1  --iterations=1000 -gpu 1 
	# echo $CAFFE_ROOT/build/tools/caffe test --model=$model --weights=$filename2  --iterations=1000 -gpu 1 
	# echo $CAFFE_ROOT/build/tools/caffe test --model=$model --weights=$filename3  --iterations=1000 -gpu 2 
	# echo $CAFFE_ROOT/build/tools/caffe test --model=$model --weights=$filename4  --iterations=1000 -gpu 3 
	./build/tools/caffe test --model=$model --weights=$filename1  --iterations=1000 -gpu 0 >$tmp1 2>&1  &
	./build/tools/caffe test --model=$model --weights=$filename2  --iterations=1000 -gpu 1 >$tmp2 2>&1  &
	./build/tools/caffe test --model=$model --weights=$filename3  --iterations=1000 -gpu 2 >$tmp3 2>&1  &
	./build/tools/caffe test --model=$model --weights=$filename4  --iterations=1000 -gpu 3 >$tmp4 2>&1  &
	echo "waiting..."
	wait
	echo "done"

	result1=$result1", "`grep -nrsI '] accuracy_top1 = ' $tmp1 | awk '{print $7 }'`
	result2=$result2", "`grep -nrsI '] accuracy_top1 = ' $tmp2 | awk '{print $7 }'`
	result3=$result3", "`grep -nrsI '] accuracy_top1 = ' $tmp3 | awk '{print $7 }'`
	result4=$result4", "`grep -nrsI '] accuracy_top1 = ' $tmp4 | awk '{print $7 }'`

	result1=$result1", "`grep -nrsI '] accuracy_top5 = ' $tmp1 | awk '{print $7 }'`
	result2=$result2", "`grep -nrsI '] accuracy_top5 = ' $tmp2 | awk '{print $7 }'`
	result3=$result3", "`grep -nrsI '] accuracy_top5 = ' $tmp3 | awk '{print $7 }'`
	result4=$result4", "`grep -nrsI '] accuracy_top5 = ' $tmp4 | awk '{print $7 }'`

	echo $result1
	echo $result2
	echo $result3
	echo $result4
	
	echo $result1 >> $output
	echo $result2 >> $output
	echo $result3 >> $output
	echo $result4 >> $output

done

sort $output  > $tmp1 
cat $tmp1> $output 

rm -rf $tmp1
rm -rf $tmp2
rm -rf $tmp3
rm -rf $tmp4
