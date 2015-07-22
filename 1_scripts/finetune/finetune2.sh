
python finetune_iterative.py --device-id=2 6 4 --update=rmsprop \
 --finetune-codebook-iters=10 --accumulate-diff-iters=3 \
 2>finetune_log/6_4_iterative_stochastic_10co_3ac.log
