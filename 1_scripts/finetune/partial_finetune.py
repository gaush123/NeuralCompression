import sys, os
from utils import analyze_log, get_accuracy

settings_str = ['8 8 8 6 6 2 2 3',
'8 8 8 6 5 2 2 3',
'8 8 6 6 5 2 2 3']

settings = map(lambda x:map(lambda y:int(y), x.split()), settings_str)
print settings

f = open('tmp/finetune_res.csv','a+')
for idx, setting in enumerate(settings):
    os.system('python finetune_solver.py %d %d %d %d %d %d %d %d --device-id=1 2>tmp/finetune_%d.log'%(tuple(setting + [idx])))
    filename = 'tmp/finetune_%d.log'%idx
    original_top1, original_top5, high_top1, high_top5 = get_accuracy(filename)
    f.write('%s, %f, %f, %f, %f\n'%(settings_str[idx], original_top1, original_top5, high_top1, high_top5))
