
from utils import analyze_log, get_accuracy

import os
def test(bits=[6, 5], option='alexnet', file=''):
    conv_bits = bits[0]
    fc_bits = bits[1]
    if len(bits) == 2:
        os.system('python finetune_solver.py %d %d --device-id=0 --network=%s 2>tmp/%d_%d.log' % (conv_bits, fc_bits, option, conv_bits, fc_bits))
    elif len(bits) == 8:
        os.system('python finetune_solver.py %d %d %d %d %d %d %d %d --device-id=0 --network=%s 2>tmp/%d_%d.log' % (tuple(bits) +  (option, conv_bits, fc_bits)))
        

    filename = 'tmp/%d_%d.log' % (conv_bits, fc_bits)
    original_top1, original_top5, high_top1, high_top5 = get_accuracy(filename)

    with open(file, 'a+') as f:
        f.write('%d, %d, %f, %f, %f, %f\n' % (conv_bits, fc_bits, original_top1, original_top5, high_top1, high_top5))

file_t = 'tmp/results_alexnet'
test([4, 10], file=file_t)
test([8, 8], file=file_t)
test([8,6], file=file_t)
test([6,4], file=file_t)
test([6,3], file=file_t)
test([5,2], file=file_t)
