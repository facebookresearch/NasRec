#!/usr/bin/sh
#SBATCH --job-name=train-supernet-xlarge-binomial-default-lr0.120-bs128
#SBATCH --cpus-per-task 8
#SBATCH --gpus-per-task 1
#SBATCH --mem 64g
#SBATCH -e avazu-xlarge-supernet-lr0.12-bs512.err
#SBATCH -o avazu-xlarge-supernet-lr0.12-bs512.out
#SBATCH --partition athena-mini
#SBATCH --account tz86

LR=0.12
WD=0

python -u nasrec/train_supernet.py \
  --learning_rate ${LR} \
  --gpu 0 \
  --root_dir ./data/avazu_kaggle_autoctr \
  --train_batch_size 512 \
  --strategy default \
  --anypath_choice binomial-0.5 \
  --logging_dir ./www-test/avazu_1shot/avazu-supernet-default-binomial_0.5-xlarge_lr${LR} \
  --test_batch_size 4096 \
  --use_layernorm 1 \
  --supernet_training_step 12500 \
  --config xlarge \
  --num_blocks 7 \
  --num_epochs 1 \
  --test_interval 2000 \
  --wd ${WD} \
  --train_limit 32343176 \
  --test_limit 4042896  \
  --dataset avazu