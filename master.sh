# 把 eth0 替换成你们局域网真正用的接口名：ip a | grep state UP
export GLOO_SOCKET_IFNAME=enp4s0      # 供 ProcessGroupGloo 建 TCP
export NCCL_SOCKET_IFNAME=enp4s0      # NCCL 同理，顺带加速
export TP_SOCKET_IFNAME=enp4s0        # TensorPipe RPC 也别用回环


torchrun \
  --nnodes 2 \
  --nproc_per_node 1 \
  --node_rank 0 \
  --master_addr 192.168.1.73 \
  --master_port 29500 \
  resnet_pipeline.py \
  --mode pipeline \
  --epochs 10 \
  --batch 128 \
  --chunks 4 \