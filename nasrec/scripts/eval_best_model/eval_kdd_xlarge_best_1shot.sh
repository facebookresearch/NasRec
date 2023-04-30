LR=0.16
WD=0

python -u nasrec/main_train.py \
    --root_dir ./data/kdd_kaggle_autoctr/ \
    --net supernet-config \
    --supernet_config nasrec/configs/kdd/ea_kdd_kaggle_xlarge_best_1shot.json \
    --num_epochs 1 \
    --learning_rate $LR \
    --train_batch_size 512 \
    --wd $WD \
    --logging_dir ./experiments-www-repro/best_models/kdd_xlarge_best_1shot_lr${LR}_wd${WD} \
    --gpu 0 \
    --test_interval 20000 \
    --dataset kdd --train_limit 134675194 --test_limit 14963910