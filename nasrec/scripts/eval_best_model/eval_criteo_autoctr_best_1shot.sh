LR=0.1
WD=0

python -u nasrec/main_train.py \
    --root_dir ./data/criteo_kaggle_autoctr/ \
    --net supernet-config \
    --supernet_config nasrec/configs/criteo/ea_criteo_kaggle_autoctr_best_1shot.json \
    --num_epochs 1 \
    --learning_rate $LR \
    --train_batch_size 256 \
    --wd $WD \
    --logging_dir ./experiments-www-repro/best_models/criteo_autoctr_best_1shot_lr${LR}_wd${WD} \
    --gpu 0 \
    --test_interval 10000