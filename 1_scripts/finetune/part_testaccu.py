
import sys
from utils import analyze_log, get_accuracy
sys.path.insert(0, '/home/maohz12/pruning/1_scripts/kmeans_quantize')
from kmeans import test_quantize_accu, caffe
caffe.set_device(0)
file_t = 'tmp/results_alexnet_onlyquantize'

settings = [[10,   5],
[9, 3],
[8, 8, 8, 6, 6, 2, 2, 3]]   
'''
[[8  ,4  ],
[9  ,4  ],
[10 ,4  ],
[11 ,4  ],
[8  ,5  ],
[9  ,5  ],
[10 ,5  ],
[11 ,5  ],
[9  ,6  ],
[10 ,6  ],
[11 ,6  ],
[8  ,7  ],
[9  ,7  ],
[10 ,7  ],
[11 ,7  ],
[8  ,8  ],
[9  ,8  ],
[10 ,8  ],
[11 ,8  ],
[9  ,9  ],
[10 ,9  ],
[11 ,9  ],
[10 ,10 ],
[11 ,10 ],
[7  ,3  ],
[8  ,3  ],
[9  ,3  ],
[10 ,3  ],
[11 ,3  ],
[5  ,2  ],
[4  ,2  ],
[3  ,2  ],
[2  ,2  ],
[6 ,6, 6, 6 ,6  ,2 ,2 ,2  ],        
[6 ,6, 6, 10, 10, 2, 3, 3 ],    
[8 ,8, 8, 6 ,6  ,2 ,2 ,3     ],
[8 ,8, 8, 6 ,5  ,2 ,2 ,3     ],
[8 ,8, 6, 6 ,5  ,2 ,2 ,3    ]]
'''

f = open(file_t, 'a+') 
for setting in settings:
    top_1, top_5 = test_quantize_accu(setting)
    f.write('%f, %f\n' % (top_1, top_5))

