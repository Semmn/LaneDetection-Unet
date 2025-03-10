val_batch_size=32
epoch=100
start_epoch=0
local_rank=0
rank=0
ds_config_path='/work/train/CULane/segmentation/ds_config.json'
project_name='convnext_unet_4x/culane'
weight_dir='/work/checkpoints'
plot_dir='/work/plots'
save_step=10
save_plot_step=5
ckpt_id='-1' # only valid when start_epoch is not 0. set this to start_epoch

# to select what gpus to use, set localhost:n (for example, localhost:0,1,2,3 use 4 gpus)
deepspeed --num_gpus=4 /work/train/CULane/segmentation/deepspeed.py --include localhost:0,1,2,3 \
    --val_batch_size $val_batch_size \
    --epoch $epoch \
    --start_epoch $start_epoch \
    --local-rank $local_rank \
    --rank $rank \
    --ds_config_path $ds_config_path \
    --project_name $project_name \
    --weight_dir $weight_dir \
    --plot_dir $plot_dir \
    --save_step $save_step \
    --save_plot_step $save_plot_step \
    --ckpt_id $ckpt_id