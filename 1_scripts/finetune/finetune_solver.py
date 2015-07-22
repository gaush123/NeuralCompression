from utils import *


codebook = kmeans_net(net, total_layers, num_c)

codeDict, maskCode = quantize_net_with_dict(net, total_layers, codebook)

print "================3 Perform fintuning=============="


for i in xrange(2500):
    solver.step(1)
    update_codebook_net(net, codebook, codeDict, maskCode, args=args, update_layers=update_layers)
    if i%20 == 0:
        print "Iters:%d"%i
