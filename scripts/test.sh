DATA_PATH="/kaggle/input/tiny-imagenet-challenge/TinyImageNet/train/0"
EMA_PATH="/kaggle/working/checkpoint/edm_imagenet64_ema.pt"

if [ -f $EMA_PATH]; then
    echo "chekck point for moving avarage model has been download"
else
    echo "downloading"
    wget -O $EMA_PATH "https://openaipublic.blob.core.windows.net/consistency/edm_imagenet64_ema.pt"
fi

mpiexec --allow-run-as-root -n 8 python cm_train.py --training_mode consistency_distillation --target_ema_mode fixed --start_ema 0.95 --scale_mode fixed --start_scales 40 --total_training_steps 600000 --loss_norm pseudo_huber --lr_anneal_steps 0 --teacher_model_path $EMA_PATH --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 2048 --image_size 64 --lr 0.000008 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir $PATH