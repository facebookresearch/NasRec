CKPT_PATH=./www-test-repro/ablations-paths/criteo-supernet-single-path-binomial_0.5-xlarge-lr0.12/supernet_7blocks_layernorm1_single-path_lr0.12_supernetwarmup_10000/supernet_checkpoint.pt
LOGGING_DIR=./www-test-repro/criteo_1shot/criteo-supernet-single-path-binomial_0.5-xlarge-ea-240gen-128pop-64sample-8childs-default-ft_lr0.04

python -u nasrec/eval_subnet_from_supernet.py \
    --root_dir ./data/criteo_kaggle_autoctr/ \
    --ea_top_k 2 \
    --ckpt_path $CKPT_PATH \
    --learning_rate 0.04 \
    --wd 0 \
    --logging_dir $LOGGING_DIR \
    --n_childs 8 \
    --n_generations 240 \
    --init_population 128 \
    --sample_size 64 \
    --method regularized-ea \
    --use_layernorm 1 \
    --max_train_steps 500 \
    --train_batch_size 512 \
    --test_batch_size 8192 \
    --num_parallel_workers 4 \
    --max_eval_steps 150 \
    --config xlarge \
    --test_only_at_last_step 1 \
    --finetune_whole_supernet 0