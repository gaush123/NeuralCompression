from utils import *

use_stochastic = args.use_stochastic
co_iters = args.co_iters
ac_iters = args.ac_iters
start_time = time.time()

codebook = kmeans_net(net, total_layers, num_c)

codeDict, maskCode = quantize_net_with_dict(net, total_layers, codebook)

for i in xrange(2500):
    if (i % (co_iters + ac_iters) == 0 and i > 0):
        codeDict, maskCode = quantize_net_with_dict(net, total_layers, codebook, use_stochastic, timing=args.timing)

    solver.step(1)
    if (i % (co_iters + ac_iters) < co_iters):
        update_codebook_net(net, codebook, codeDict, maskCode, args=args, update_layers=update_layers)

    if args.timing:
        print "Iter:%d, Time cost:%f" % (i, time.time() - start_time)
