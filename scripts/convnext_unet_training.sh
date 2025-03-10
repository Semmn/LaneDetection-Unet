batch_size=64
val_batch_size=64

model_config='/work/train/segmentation/model_config/unet.yaml'
train_config='/work/train/segmentation/train_config/segmentation_culane.yaml'

project_name='convnext_unet_4x/culane/pretrained'
weight_dir='/work/checkpoints'
plot_dir='/work/plots'
save_step=5
save_plot_step=5
load_weight_path='None'
pretrained_weight_path='/work/checkpoints/convnext_unet_4x/multi_data/self_supervised/2025_02_17_150135/weights/2025_02_22_202021_epoch40.pth'

rank=0
local_rank=0
num_workers=16
# gpu_ids must be presented as string delimited by ','
gpu_ids='0,1,2,3,4,5,6,7' # in case of using all gpus, gpu_ids='0,1,2,3,4,5,6,7'
world_size=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 /work/train/segmentation/multigpu.py \
    --batch_size $batch_size \
    --val_batch_size $val_batch_size \
    --model_config $model_config \
    --train_config $train_config \
    --project_name $project_name \
    --weight_dir $weight_dir \
    --plot_dir $plot_dir \
    --save_step $save_step \
    --save_plot_step $save_plot_step \
    --load_weight_path $load_weight_path \
    --pretrained_weight_path $pretrained_weight_path \
    --rank $rank \
    --local-rank $local_rank \
    --num_workers $num_workers \
    --gpu_ids $gpu_ids \
    --world_size $world_size