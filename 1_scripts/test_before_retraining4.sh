#!/bin/bash
# no argument required
# thresh_list=(0.46 0.72 1.05 1.27 1.44 1.58 1.70 1.81 1.91 2.00)
# thresh_list=(0 0.25 0.69 1.06 1.35 1.59 1.80 1.99 2.16 2.32)
# thresh_list=(0.08 2.32 2.64 3.38)
# thresh_list=(0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.30 0.31 0.32 0.33 0.34)
# thresh_list=(0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 )
# thresh_list=(1.0 1.01 1.02 1.03 1.04 1.05 1.06 1.07 1.08 1.09 1.1)
thresh_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0)
thresh_list=(1.7 1.8 1.9 2.0)
folder=L2
suffix="fc7"
suffix_2="afterConv8x_"
# suffix="all"
# suffix_2="layerwise_" 
# suffix_2="alex_pruned_"
# suffix_2='alex_pruned_afterConv_'
# suffix_2="afterConv7x_"


model="$CAFFE_ROOT/3_prototxt_solver/$folder/train_val.prototxt"
output="$CAFFE_ROOT/2_results/$folder/acc_before_retrain_$suffix_2$suffix.csv"
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
	
# 	data1=`printf "%.2f" $data1`
# 	data2=`printf "%.2f" $data2`
# 	data3=`printf "%.2f" $data3`
# 	data4=`printf "%.2f" $data4`

	result1=$data1
	result2=$data2
	result3=$data3
	result4=$data4
	filename1=$filename_prefix$data1"_$suffix.caffemodel"		
	filename2=$filename_prefix$data2"_$suffix.caffemodel"		
	filename3=$filename_prefix$data3"_$suffix.caffemodel"		
	filename4=$filename_prefix$data4"_$suffix.caffemodel"		
# 	echo $result1
# 	echo $result2
# 	echo $result3
# 	echo $result4
	echo $filename1
	echo $filename2
	echo $filename3
	echo $filename4
	rm -rf $tmp1
	rm -rf $tmp2
	rm -rf $tmp3
	rm -rf $tmp4
	echo $CAFFE_ROOT/build/tools/caffe test --model=$model --weights=$filename1  --iterations=1000 -gpu 0 
	echo $CAFFE_ROOT/build/tools/caffe test --model=$model --weights=$filename2  --iterations=1000 -gpu 1 
	echo $CAFFE_ROOT/build/tools/caffe test --model=$model --weights=$filename3  --iterations=1000 -gpu 2 
	echo $CAFFE_ROOT/build/tools/caffe test --model=$model --weights=$filename4  --iterations=1000 -gpu 3 
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
cat $output

# sort $output  > $tmp1 
# cat $tmp1> $output 

rm -rf $tmp1
rm -rf $tmp2
rm -rf $tmp3
rm -rf $tmp4
