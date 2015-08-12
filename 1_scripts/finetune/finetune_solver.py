
from utils import *
from init import *

codebook = kmeans_net(net, total_layers, num_c, method=args.kmeans_method, compress=args.kmeans_compress)

codeDict, maskCode = quantize_net_with_dict(net, total_layers, codebook)

print "================3 Perform fintuning=============="

start_time = time.time()

if args.network=='network':
    cycles = 3000
else:
    cycles = 1000
for i in xrange(cycles):
    solver.step(1)
    if (i + 1) % 1000 == 0 and args.normalize_flag:
        args.lr /= 10
    update_codebook_net(net, codebook, codeDict, maskCode, args=args, update_layers=update_layers)
    if args.timing:
        print "Iter:%d, Time cost:%f" % (i, time.time() - start_time)
