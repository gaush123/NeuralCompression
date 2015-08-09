import sys, os
from utils import analyze_log, get_accuracy

settings = [[6  ,3  ],
[7  ,3  ],
[9  ,3  ],
[11 ,3  ],
[11 ,2  ],
[8  ,2  ],
[5  ,2  ],
[4  ,2  ]]
# settings = map(lambda x:map(lambda y:int(y), x.split()), settings_str)
print settings

f = open('tmp/finetune_quantize_only.csv','w')
for idx, setting in enumerate(settings):
    os.system('python finetune_solver.py %d %d --network=alexnet_original --device-id=0 2>tmp/finetune_%d.log'%(tuple(setting + [idx])))
    filename = 'tmp/finetune_%d.log'%idx
    original_top1, original_top5, high_top1, high_top5 = get_accuracy(filename)
    f.write('%f, %f, %f, %f\n'%(original_top1, original_top5, high_top1, high_top5))

settings = [[10,8, 8, 8, 8, 2, 3, 2    ],  
[6 ,6, 6, 10, 10, 2, 3, 3  ],   
[10,10, 8, 10, 10, 3, 2, 3 ],    
[8 ,8, 8, 6 ,6  ,2 ,2 ,3   ],
[8 ,8, 8, 6 ,5  ,2 ,2 ,3   ],
[10,10, 8, 8 ,6  ,2 ,2 ,3   ],
[8 ,8, 6, 6 ,5  ,2 ,2 ,3   ]]

for idx, setting in enumerate(settings):
    os.system('python finetune_solver.py %d %d %d %d %d %d %d %d --network=alexnet_original --device-id=0 2>tmp/finetune_%d.log'%(tuple(setting + [idx])))
    filename = 'tmp/finetune_%d.log'%idx
    original_top1, original_top5, high_top1, high_top5 = get_accuracy(filename)
    f.write('%f, %f, %f, %f\n'%(original_top1, original_top5, high_top1, high_top5))

