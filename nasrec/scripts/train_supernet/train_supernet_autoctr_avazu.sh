LR=0.12
WD=0

python -u nasrec/train_supernet.py \
  --learning_rate ${LR} \
  --gpu 0 \
  --root_dir ./data/avazu_kaggle_autoctr \
  --train_batch_size 512 \
  --strategy default \
  --anypath_choice binomial-0.5 \
  --logging_dir ./www-test/avazu_1shot/avazu-supernet-default-binomial_0.5-autoctr_lr${LR} \
  --test_batch_size 4096 \
  --use_layernorm 1 \
  --supernet_training_step 12500 \
  --config autoctr \
  --num_blocks 7 \
  --num_epochs 1 \
  --test_interval 2000 \
  --wd ${WD} \
  --train_limit 32343176 \
  --test_limit 4042896  \
  --dataset avazu 