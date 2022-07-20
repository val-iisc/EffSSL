export MASTER_PORT=99990
export WORLD_SIZE=4

DATASET_PATH='/path/to/imagenet'
PRETRAIN_EXP_PATH='/path/to/pretrain_experiment'
EVAL_EXP_PATH='/path/to/eval_experiment'

python -m torch.distributed.launch --nproc_per_node=$WORLD_SIZE --master_port=$MASTER_PORT main_swav.py \
--arch resnet50 \
--data_path $DATASET_PATH \
--nmb_crops 2 6 \
--size_crops 224 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--nmb_prototypes 3000 \
--queue_length 0 \
--epochs 50 \
--batch_size 256 \
--base_lr 1.2 \
--final_lr 0.0012 \
--freeze_prototypes_niters 313 \
--wd 0.000001 \
--warmup_epochs 10 \
--start_warmup 0.3 \
--use_fp16 true \
--dump_path $PRETRAIN_EXP_PATH \
--checkpoint_freq 1

python -m torch.distributed.launch --nproc_per_node=$WORLD_SIZE --master_port=$MASTER_PORT eval_linear.py \
--data_path $DATASET_PATH \
--pretrained $PRETRAIN_EXP_PATH/checkpoint.pth.tar \
--epoch 25 \
--batch_size 64 \
--lr 0.3 \
--dump_path $EVAL_EXP_PATH
