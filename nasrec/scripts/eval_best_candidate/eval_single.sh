LR=0.12
ID=5

CUDA_VISIBLE_DEVICES=7 python -u nasrec/main_train.py --root_dir ./data/criteo_kaggle_autoctr/ \
        --net supernet-config \
        --supernet_config ea-criteo-kaggle-xlarge-best-1shot/best_config_${ID}.json \
        --num_epochs 1 \
        --learning_rate ${LR} \
        --train_batch_size 512 \
        --wd 0 \
        --logging_dir ./www-test/ea-best-criteo-kaggle-config-${ID} \
        --gpu 0 \
        --test_interval 5000 \
        --dataset criteo-kaggle