#!/bin/bash
#need two arguments: begin, end

suffix=""
model="$CAFFE_ROOT/3_prototxt_solver/L2/train_val0.46.prototxt"
# model="$CAFFE_ROOT/3_prototxt_solver/L1_3/train_val.prototxt"
filename_prefix="$CAFFE_ROOT/4_model_checkpoint/2_after_retrain/L2/prune1.44_iter_"
# filename_prefix="/cnn/caffemodel/caffe_alexnet_train1_iter_"

if [ "$1" -gt "$2" ]; then
    let start=$2
    let end=$1
else 
    let start=$1
    let end=$2
fi

output=result_$start_$end.csv
tmp1=.`date +"%T.%3N"`.tmp
tmp2=.`date +"%T.%3N"`.tmp
tmp3=.`date +"%T.%3N"`.tmp
tmp4=.`date +"%T.%3N"`.tmp

for i in $(seq $end -20000 $start);
do    
    let data1=$i
    let data2=$data1-5000
    let data3=$data2-5000
    let data4=$data3-5000
    
#     data1=`printf "%.2f" $data1`
#     data2=`printf "%.2f" $data2`
#     data3=`printf "%.2f" $data3`
#     data4=`printf "%.2f" $data4`

    result1=$data1
    result2=$data2
    result3=$data3
    result4=$data4
    echo $data1
    echo $data2
    echo $data3
    echo $data4
    filename1=$filename_prefix$data1"$suffix.caffemodel"        
    filename2=$filename_prefix$data2"$suffix.caffemodel"        
    filename3=$filename_prefix$data3"$suffix.caffemodel"        
    filename4=$filename_prefix$data4"$suffix.caffemodel"        
    echo $filename1
    echo $filename2
    echo $filename3
    echo $filename4
    rm -rf $tmp1
    rm -rf $tmp2
    rm -rf $tmp3
    rm -rf $tmp4
    echo ./build/tools/caffe test --model=$model --weights=$filename1  --iterations=1000 -gpu 0 
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
