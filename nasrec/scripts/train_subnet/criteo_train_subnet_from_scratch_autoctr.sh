LR=0.12

python -u nasrec/eval_subnet_from_scratch.py \
  --learning_rate ${LR} \
  --gpu 0 \
  --root_dir ./data/criteo_kaggle_autoctr \
  --train_batch_size 1024 \
  --logging_dir ./www-test/criteo-100subnets-autoctr-b0 \
  --test_batch_size 8000 \
  --use_layernorm 1 \
  --config autoctr \
  --num_blocks 7 \
  --num_subnets 100