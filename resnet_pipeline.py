import os, time, argparse, itertools, warnings
from typing import List

import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
import torch.distributed.optim as dist_optim
from torchvision.models import resnet18
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

START_T = time.time()


def log(msg, rank):
    print(f"[{time.time()-START_T:6.3f}s] rank{rank}: {msg}", flush=True)


# ---------- 数据 ----------
def synthetic_loader(batch: int, total_batches: int, device: str):
    for _ in range(total_batches):
        x = torch.randn(batch, 3, 224, 224, device=device)
        y = torch.randint(0, 1000, (batch,), device=device)
        yield x, y


# ---------- 模型切分 ----------
def split_resnet18() -> List[nn.Module]:
    full = resnet18(num_classes=1000)
    stage0 = nn.Sequential(
        full.conv1, full.bn1, full.relu, full.maxpool, full.layer1, full.layer2
    )
    stage1 = nn.Sequential(
        full.layer3, full.layer4, full.avgpool, nn.Flatten(1), full.fc
    )
    return [stage0, stage1]


# ---------- 远程 Stage ----------
class Stage(nn.Module):
    def __init__(self, sub: nn.Module):
        super().__init__()
        self.device = torch.cuda.current_device()
        self.sub = sub.to(self.device)

    def forward(self, x):
        x = x.to(self.device, non_blocking=True)
        y = self.sub(x)
        return y.cpu()  # 传回 Driver / 下一段

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.sub.parameters()]


# ---------- Driver ----------
class PipelineDriver:
    def __init__(self, workers: List[str], chunks: int):
        self.workers = workers
        self.chunks = chunks

        stages = split_resnet18()
        self.stage_rrefs = [
            rpc.remote(w, Stage, args=(sub,)) for w, sub in zip(workers, stages)
        ]
        self.param_rrefs = list(
            itertools.chain.from_iterable(
                s.rpc_sync().parameter_rrefs() for s in self.stage_rrefs
            )
        )
        self.criterion = nn.CrossEntropyLoss()
        self.opt = dist_optim.DistributedOptimizer(
            torch.optim.SGD, self.param_rrefs, lr=0.05, momentum=0.9
        )

    # 异步流水一个 micro-batch
    def _run_microbatch(self, mb_x: torch.Tensor):
        fut = self.stage_rrefs[0].rpc_async().forward(mb_x.cpu())
        for nxt in self.stage_rrefs[1:]:
            fut = fut.then(lambda pf, r=nxt: r.rpc_async().forward(pf.wait()))
        return fut.then(lambda lf: lf.wait())  # 返回 Tensor

    def train_epoch(self, loader, rank):
        total_loss, total = 0.0, 0
        for step, (x, y) in enumerate(loader):
            mb_sz = x.size(0) // self.chunks
            micro_x = x.chunk(self.chunks)
            micro_y = y.chunk(self.chunks)

            with dist_autograd.context() as ctx:
                futures = [self._run_microbatch(mb) for mb in micro_x]
                outs_gpu = [f.wait().cuda() for f in futures]
                tgt_gpu = [t.cuda() for t in micro_y]

                loss = torch.stack(
                    [self.criterion(o, t) for o, t in zip(outs_gpu, tgt_gpu)]
                ).mean()

                dist_autograd.backward(ctx, [loss])
                self.opt.step(ctx)

            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            if step % 10 == 0:
                log(f"step {step}  avg_loss={total_loss/total:.3f}", rank)
        return total_loss / total


# ---------- 单机基准 ----------
def single_gpu_train(args):
    torch.cuda.set_device(0)
    device = "cuda:0"
    model = resnet18(num_classes=1000).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    crit = nn.CrossEntropyLoss()

    batches = args.iters
    tic = time.time()
    for ep in range(args.epochs):
        loader = synthetic_loader(args.batch, batches, device)
        tot, tot_loss = 0, 0.0
        for stp, (x, y) in enumerate(loader):
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            model.zero_grad(set_to_none=True)

            tot_loss += loss.item() * x.size(0)
            tot += x.size(0)
            if stp % 10 == 0:
                log(f"single epoch{ep} step{stp}", 0)
        print(f"epoch {ep} avg_loss={tot_loss/tot:.3f}")
    elapsed = time.time() - tic
    print(f"=== single-gpu total {elapsed:.2f}s ===")
    return elapsed


# ---------- 可视化 ----------
def visualize(t_single, t_pipe, gpus):
    acc = t_single / t_pipe
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["single"], [t_single])
    ax.bar([f"{gpus}-gpu\npipe"], [t_pipe])
    ax.set_ylabel("seconds  (lower is better)")
    ax.set_title(f"speed-up {acc:.2f}×")
    for i, v in enumerate([t_single, t_pipe]):
        ax.text(i, v * 1.02, f"{v:.1f}s", ha="center")
    plt.tight_layout()
    plt.savefig("speedup.png")
    print("saved speedup.png")


# ---------- Pipeline 主程 ----------
def run_pipeline(args):
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())

    rpc_opts = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)
    rpc.init_rpc(
        f"worker{rank}", rank=rank, world_size=world, rpc_backend_options=rpc_opts
    )
    log("rpc initialized", rank)

    if rank == 0:
        workers = [f"worker{r}" for r in range(world)]
        driver = PipelineDriver(workers, args.chunks)

        loader = synthetic_loader(args.batch, args.iters, device="cuda:0")
        tic = time.time()
        for ep in range(args.epochs):
            loss = driver.train_epoch(loader, rank)
            log(f"epoch {ep} done  loss={loss:.3f}", rank)
        t_pipe = time.time() - tic
        log(f"=== pipeline total {t_pipe:.2f}s ===", rank)

        if args.single_time:
            visualize(args.single_time, t_pipe, world)
    rpc.shutdown()


# ---------- CLI ----------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--mode", choices=["single", "pipeline"], default="single")
    pa.add_argument("--epochs", type=int, default=2)
    pa.add_argument("--batch", type=int, default=64)
    pa.add_argument("--chunks", type=int, default=4)
    pa.add_argument("--iters", type=int, default=40)
    pa.add_argument("--single_time", type=float, default=None)
    args = pa.parse_args()

    if args.mode == "single":
        t = single_gpu_train(args)
        print("将该用时填入 --single_time 以绘制加速图")
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
