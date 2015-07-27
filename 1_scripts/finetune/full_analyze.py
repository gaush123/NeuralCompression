import sys, os

from utils import analyze_log, get_accuracy

sys.path.insert(0, '/home/maohz12/pruning/1_scripts/analyze_statistics')
from calculate_compress_rate import get_results


bits_range = [2, 10]
os.system('mkdir full_log')
for fc_bits in range(bits_range[0], bits_range[1] + 1):
    '''
    for conv_bits in range(fc_bits, bits_range[1] + 1):
        os.system('python 1_scripts/finetune/finetune_solver.py %d %d --device_id=0 2>full_log/%d_%d.log'%((conv_bits, fc_bits) * 2))

    '''
    for conv_bits in range(fc_bits, bits_range[1] + 2, 2):
        os.system('python 1_scripts/finetune/finetune_solver.py %d %d --device-id=0 2>full_log/%d_%d.log & \
                   python 1_scripts/finetune/finetune_solver.py %d %d --device-id=2 2>full_log/%d_%d.log' % ((conv_bits, fc_bits) * 2 + (conv_bits + 1, fc_bits) * 2))

f = open('full_results', 'a+')
f.write('conv, fc, loc, compress rate, top1 before finetune, top5 before finetune, top1 after finetune, top5 after finetune\n')
for fc_bits in range(bits_range[0], bits_range[1] + 1):
    for conv_bits in range(fc_bits, bits_range[1] + 2):
        filename = 'full_log/%d_%d.log' % (conv_bits, fc_bits)
        if os.path.isfile(filename):
            original_top1, original_top5, high_top1, high_top5 = get_accuracy(filename)
            location_bits, compress_rate = get_results([conv_bits, fc_bits])
            f.write('%d, %d, %d, %f, %f, %f, %f, %f\n' % (conv_bits, fc_bits, location_bits, compress_rate, original_top1, original_top5, high_top1, high_top5))

f.close()
