MASTER_ADDR=192.168.1.73
MASTER_PORT=29500
torchrun \
    --nnodes=2 \
    --nproc_per_node=1 \
    --node_rank=0 \
    resnet_pipeline.py \
    --mode pipeline \
    --epochs 5 \
    --batch 128 \
    --chunks 4 \
    --single_time 37.79