from utils import *

use_stochastic = args.use_stochastic
co_iters = args.co_iters
ac_iters = args.ac_iters

for i in xrange(2500):
    if (i%(co_iters + ac_iters) == 0 and i > 0):
        codeDict, maskCode = quantize_net_with_dict(net, codebook, use_stochastic)

    solver.step(1)
    if (i%(co_iters + ac_iters) < co_iters):
        update_codebook_net(net, codebook, codeDict, maskCode, args=args, update_layers=update_layers)

    if i%20 == 0:
        print "Iters:%d"%i
