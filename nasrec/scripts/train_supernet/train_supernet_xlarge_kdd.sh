LR=0.12
WD=0

python -u nasrec/train_supernet.py \
  --learning_rate ${LR} \
  --gpu 0 \
  --root_dir ./data/kdd_kaggle_autoctr \
  --train_batch_size 1024 \
  --strategy default \
  --anypath_choice binomial-0.5 \
  --logging_dir ./www-test/kdd_1shot/supernet-default-binomial_0.5-xlarge_lr${LR} \
  --test_batch_size 4096 \
  --use_layernorm 1 \
  --supernet_training_step 20000 \
  --config xlarge \
  --num_blocks 7 \
  --num_epochs 1 \
  --test_interval 2000 \
  --wd ${WD} \
  --train_limit 119711284 \
  --test_limit 14963910 \
  --dataset kdd 