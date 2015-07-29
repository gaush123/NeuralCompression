import sys, os

from utils import analyze_log, get_accuracy

sys.path.insert(0, '/home/maohz12/pruning/1_scripts/analyze_statistics')
sys.path.insert(0, '/home/maohz12/pruning/1_scripts/kmeans_quantize')
from calculate_compress_rate import get_results
from kmeans import test_quantize_accu
from itertools import combinations_with_replacement as combinations
from random import randint

fc_range = [2,3]
conv_range = [3,4,6,8,10]
compress_rate_max = 0.32

f = open('tmp/partial_search2.csv','a+')
'''
for conv_setting in combinations(conv_range, 3):
    conv_bits = [conv_setting[0]] * 2 + [conv_setting[1]] + [conv_setting[2]] * 2
    for fc_bits in combinations(fc_range, 3):
        if randint(0,2) > 0:
            continue
        location_bits, compress_rate = get_results(conv_bits + list(fc_bits))
        if compress_rate > compress_rate_max:
            continue

        try:
            top_1, top_5 = test_quantize_accu(conv_bits + list(fc_bits))
        except Exception as e:
            print e
            f.write('%d %d %d %d %d %d %d %d, %f, %f, %f\n'%(tuple(conv_bits) + fc_bits + (compress_rate, top_1, top_5)))
            continue
        f.write('%d %d %d %d %d %d %d %d, %f, %f, %f\n'%(tuple(conv_bits) + fc_bits + (compress_rate, top_1, top_5)))
        f.flush()



fc_range = [1,2]
conv_range = [2, 3,4,6]
compress_rate_max = 0.30

for conv_setting in combinations(conv_range, 3):
    conv_bits = [conv_setting[0]] * 2 + [conv_setting[1]] + [conv_setting[2]] * 2
    for fc_bits in combinations(fc_range, 3):
        if randint(0,2) > 0:
            continue
        location_bits, compress_rate = get_results(conv_bits + list(fc_bits))
        if compress_rate > compress_rate_max:
            continue

        try:
            top_1, top_5 = test_quantize_accu(conv_bits + list(fc_bits))
        except Exception as e:
            print e
            f.write('%d %d %d %d %d %d %d %d, %f, %f, %f\n'%(tuple(conv_bits) + fc_bits + (compress_rate, top_1, top_5)))
            continue
        f.write('%d %d %d %d %d %d %d %d, %f, %f, %f\n'%(tuple(conv_bits) + fc_bits + (compress_rate, top_1, top_5)))
        f.flush()
'''
extra_settings = [
[8,8,8,6,6,2,2,2],
[8,8,8,6,6,2,2,3],
[10,8,8,6,6,2,2,2],
[8,8,6,6,6,2,2,3],
[8,8,8,8,6,2,2,2],
[8,8,8,6,5,2,2,3],
[8,8,6,6,5,2,2,3]
]

compress_rate_max = 0.030
compress_rate_min = 0.028
for setting in extra_settings:
    location_bits, compress_rate = get_results(setting)
    if compress_rate > compress_rate_max or compress_rate < compress_rate_min:
        continue

    try:
        top_1, top_5 = test_quantize_accu(setting)
    except Exception as e:
        print e
        f.write('%d %d %d %d %d %d %d %d, %f, %f, %f\n'%(tuple(setting) + (compress_rate, top_1, top_5)))
        continue
    f.write('%d %d %d %d %d %d %d %d, %f, %f, %f\n'%(tuple(setting) + (compress_rate, top_1, top_5)))
    f.flush()
