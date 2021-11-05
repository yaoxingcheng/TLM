TASK=sciie
SCALE=small

if [[ $TASK == "imdb" ]]
then
MAXLEN=512
else 
MAXLEN=128
fi

accelerate launch --config_file ./accelerate_config/example_config.yaml src/run.py \
    --max_train_steps 0 \
    --preprocessing_num_workers 32 \
    --max_length $MAXLEN \
    --pad_to_max_length \
    --model_name_or_path yxchar/tlm-${TASK}-${SCALE}-scale \
    --config_dir yxchar/tlm-${TASK}-${SCALE}-scale \
    --per_device_eval_batch_size 16 \
    --task_name $TASK