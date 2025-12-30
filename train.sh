ngpu=4 accelerate launch --config_file config/default.yaml \
    --num_processes=${ngpu} \
    train.py \
        --pretrained_model_name_or_path sd3.5_medium \
        --data_root demo \
        --work_dir demo \
        --train_file emo/index_.jsonl \
        --vectors_path vectors/demo \
        --num_cls 18