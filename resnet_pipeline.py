import os, sys, time, math, argparse, itertools, random, warnings
from typing import List
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
import torch.distributed.optim as dist_optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18
import matplotlib.pyplot as plt

# ============ 公共工具 ============
def log(msg, rank):
    elapsed = f"{time.time()-START_T:.3f}s"
    print(f"[{elapsed}] rank{rank}: {msg}", flush=True)

def synthetic_loader(batch, total_batches, device):
    """用随机数据代替真实数据，避免下载数据集耗时。"""
    for _ in range(total_batches):
        x = torch.randn(batch, 3, 224, 224, device=device)
        y = torch.randint(0, 1000, (batch,), device=device)
        yield x, y

def split_resnet18() -> List[nn.Module]:
    """按层拆 ResNet18 -> 两段（可自行增减）"""
    full = resnet18(num_classes=1000)
    # 第一段：conv1 ~ layer2
    stage0 = nn.Sequential(
        full.conv1, full.bn1, full.relu, full.maxpool,
        full.layer1, full.layer2
    )
    # 第二段：layer3+layer4+fc
    stage1 = nn.Sequential(
        full.layer3, full.layer4, full.avgpool,
        nn.Flatten(1), full.fc
    )
    return [stage0, stage1]

# ============ Stage 定义 ============
class Stage(nn.Module):
    """每个 rank 上真正跑的子模块"""
    def __init__(self, submod: nn.Module):
        super().__init__()
        self.sub = submod.to(torch.cuda.current_device())

    def forward(self, x):
        return self.sub(x)

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.sub.parameters()]

# ============ Pipeline Driver ============
class PipelineDriver:
    """
    rank0 作为 driver：
    - 拆 batch 为 chunks
    - 串起远程阶段，形成流水线
    """
    def __init__(self, workers: List[str], chunks: int):
        self.workers = workers            # worker0, worker1, ...
        self.chunks = chunks
        # 远程创建 Stage
        self.stage_rrefs = []
        stages = split_resnet18()
        for w, sub in zip(workers, stages):
            self.stage_rrefs.append(
                rpc.remote(w, Stage, args=(sub,))
            )
        # 收集全部参数 RRef，供分布式优化器使用
        self.param_rrefs = list(
            itertools.chain.from_iterable(
                s.rpc_sync().parameter_rrefs() for s in self.stage_rrefs
            )
        )
        self.criterion = nn.CrossEntropyLoss()
        self.opt = dist_optim.DistributedOptimizer(
            torch.optim.SGD,
            self.param_rrefs,
            lr=0.05, momentum=0.9
        )

    def _run_microbatch(self, mb_x):
        """
        异步串联调用每个 stage；返回 Future
        """
        fut = self.stage_rrefs[0].rpc_async().forward(mb_x)

        # 链式 then，形成流水
        for next_stage in self.stage_rrefs[1:]:
            fut = fut.then(
                lambda y, r=next_stage: r.rpc_async().forward(y)
            )
        return fut

    def train_epoch(self, loader, rank):
        total_loss, total = 0.0, 0
        for i, (x, y) in enumerate(loader):
            micro_size = x.size(0) // self.chunks
            micro_x = x.chunk(self.chunks)
            micro_y = y.chunk(self.chunks)

            with dist_autograd.context() as ctx:
                # ---- 前向：并发发射每个 microbatch ----
                futures = []
                for mb in micro_x:
                    futures.append(self._run_microbatch(mb))

                # 同步等待全部输出
                outs = [f.wait() for f in futures]

                # ---- 计算 loss & 反向 ----
                loss = torch.stack([
                    self.criterion(o, t) for o, t in zip(outs, micro_y)
                ]).mean()

                dist_autograd.backward(ctx, [loss])
                self.opt.step(ctx)

            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            if i % 10 == 0:
                log(f"epoch progress {i}/{len(loader)} "
                    f"avg_loss={total_loss/total:.3f}", rank)
        return total_loss / total

# ============ 单 GPU 训练 ============
def single_gpu_train(args):
    torch.cuda.set_device(0)
    device = "cuda:0"
    model = resnet18(num_classes=1000).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    batches_per_epoch = args.iters
    loader = synthetic_loader(args.batch, batches_per_epoch, device)

    tic = time.time()
    for ep in range(args.epochs):
        total_loss = 0.0
        total = 0
        for i, (x, y) in enumerate(loader):
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            model.zero_grad(set_to_none=True)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            if i % 10 == 0:
                print(f"[{time.time()-tic:.2f}s] single epoch "
                      f"{ep} step {i}/{batches_per_epoch}")
        print(f"epoch {ep} avg_loss={total_loss/total:.3f}")
    elapsed = time.time() - tic
    print(f"=== single-gpu total time: {elapsed:.2f}s ===")
    return elapsed

# ============ Pipeline 主函数 ============
def run_pipeline(args):
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # worker 名称，便于 rpc(remote)
    worker_name = f"worker{rank}"
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)
    rpc.init_rpc(worker_name, rank=rank, world_size=world,
                 rpc_backend_options=options)

    log("rpc initialized", rank)

    # rank0 driver
    if rank == 0:
        # 构造流水线
        workers = [f"worker{r}" for r in range(world)]
        driver = PipelineDriver(workers, args.chunks)

        batches_per_epoch = args.iters
        loader = synthetic_loader(args.batch, batches_per_epoch,
                                  device="cuda:0")

        tic = time.time()
        for ep in range(args.epochs):
            loss = driver.train_epoch(loader, rank)
            log(f"epoch {ep} finished, loss={loss:.3f}", rank)
        elapsed = time.time() - tic
        log(f"=== pipeline total time: {elapsed:.2f}s ===", rank)

        # 可视化对比（与基准单卡时间一起画）
        if args.single_time:
            visualize(args.single_time, elapsed, world)
    # 其余 rank 仅保持 rpc 运行
    rpc.shutdown()

# ============ 可视化 ============
def visualize(t_single, t_pipeline, gpus):
    accel = t_single / t_pipeline
    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(["single"], [t_single])
    ax.bar([f"{gpus}-gpu\npipeline"], [t_pipeline])
    ax.set_ylabel("Time / s  (lower is better)")
    ax.set_title(f"Speed-up = {accel:.2f}×")
    for i,v in enumerate([t_single, t_pipeline]):
        ax.text(i, v*1.01, f"{v:.1f}s", ha='center')
    plt.tight_layout()
    plt.savefig("speedup.png")
    print("figure saved: speedup.png")

# ============ CLI ============
START_T = time.time()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pipeline", "single"], default="single")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=64,
                   help="global batch (会被 driver 均分为 chunks)")
    p.add_argument("--chunks", type=int, default=4,
                   help="micro-batch 个数")
    p.add_argument("--iters", type=int, default=40,
                   help="batches per epoch (synthetic)")
    p.add_argument("--single_time", type=float, default=None,
                   help="基准单卡耗时（由单卡跑完后手动填入即可绘图）")
    args = p.parse_args()

    if args.mode == "single":
        t = single_gpu_train(args)
        print("将此数值作为 --single_time 传给 pipeline 模式 "
              "即可绘制加速图。")
    else:
        run_pipeline(args)

if __name__ == "__main__":
    main()

"""
# ---- 多机多卡启动示例（2 节点 × 1 GPU）----
# 节点 A（master）：
MASTER_ADDR=<master_ip>
MASTER_PORT=29500
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
    resnet_pipeline.py --mode pipeline --epochs 5 --batch 128 --chunks 4

# 节点 B（worker）：
MASTER_ADDR=<master_ip>
MASTER_PORT=29500
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
    resnet_pipeline.py --mode pipeline --epochs 5 --batch 128 --chunks 4

# ---- 单机单卡基准 ----
python resnet_pipeline.py --mode single --epochs 5 --batch 128
"""