
python finetune_iterative.py --device-id=3 6 4 --update=rmsprop \
 --finetune-codebook-iters=20 --accumulate-diff-iters=10 \
 2>finetune_log/6_4_iterative_stochastic_20co_19ac.log
