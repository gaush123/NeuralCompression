from utils import *


codebook = kmeans_net(net, total_layers, num_c, method=args.kmeans_method, compress=args.kmeans_compress)

codeDict, maskCode = quantize_net_with_dict(net, total_layers, codebook)

print "================3 Perform fintuning=============="

start_time = time.time()

for i in xrange(2500):
    solver.step(1)
    update_codebook_net(net, codebook, codeDict, maskCode, args=args, update_layers=update_layers)
    if args.timing:
        print "Iter:%d, Time cost:%f"%(i, time.time() - start_time)
