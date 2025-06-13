import os, time, argparse, itertools, warnings
from typing import List

import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
import torch.distributed.optim as dist_optim
from torchvision.models import resnet18

warnings.filterwarnings("ignore", category=UserWarning)
START = time.time()


def log(m, r):
    print(f"[{time.time()-START:6.2f}s] rk{r}: {m}", flush=True)


# ---------- 数据 ----------
def fake_loader(bsz: int, steps: int, dev: str):
    for _ in range(steps):
        yield torch.randn(bsz, 3, 224, 224, device=dev), torch.randint(
            0, 1000, (bsz,), device=dev
        )


# ---------- 切两段 ----------
def split_resnet18():
    m = resnet18(num_classes=1000)
    return [
        nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2),
        nn.Sequential(m.layer3, m.layer4, m.avgpool, nn.Flatten(1), m.fc),
    ]


# ---------- 远程 Stage ----------
class Stage(nn.Module):
    def __init__(self, sub):
        super().__init__()
        self.dev = torch.cuda.current_device()
        self.sub = sub.to(self.dev)

    def forward(self, x):
        y = self.sub(x.to(self.dev, non_blocking=True))
        return y.cpu()

    def params(self):
        return [rpc.RRef(p) for p in self.sub.parameters()]


# ---------- Driver ----------
class Driver:
    def __init__(self, workers: List[str], chunks: int):
        self.chunks = chunks
        self.stg0, self.stg1 = [
            rpc.remote(w, Stage, (s,)) for w, s in zip(workers, split_resnet18())
        ]
        self.params = self.stg0.rpc_sync().params() + self.stg1.rpc_sync().params()
        self.crit = nn.CrossEntropyLoss()
        self.opt = dist_optim.DistributedOptimizer(
            torch.optim.SGD, self.params, lr=0.05, momentum=0.9
        )

    # 发到 stage-0；回调再发 stage-1；最终返回 Future[Tensor]
    def _exec_pipeline(self, x_cpu: torch.Tensor):
        f0 = self.stg0.rpc_async().forward(x_cpu)

        def _to_stage1(t):
            return self.stg1.rpc_sync().forward(t)  # 返回真正张量

        return f0.then(_to_stage1)  # 返回 Future[Tensor]

    def train_epoch(self, loader, rank):
        totl, tot = 0.0, 0
        for step, (x, y) in enumerate(loader):
            xs, ys = x.chunk(self.chunks), y.chunk(self.chunks)
            with dist_autograd.context() as ctx:
                futs = [self._exec_pipeline(m.cpu()) for m in xs]
                outs = [f.wait().cuda() for f in futs]  # 张量 OK
                tgts = [t.cuda() for t in ys]
                loss = torch.stack([self.crit(o, t) for o, t in zip(outs, tgts)]).mean()
                dist_autograd.backward(ctx, [loss])
                self.opt.step(ctx)
            totl += loss.item() * x.size(0)
            tot += x.size(0)
            if step % 10 == 0:
                log(f"step{step} loss={totl/tot:.3f}", rank)
        return totl / tot


# ---------- 单卡基准 ----------
def single(args):
    dev = "cuda:0"
    torch.cuda.set_device(0)
    net, res = resnet18().to(dev), nn.CrossEntropyLoss()
    opt = torch.optim.SGD(net.parameters(), 0.05, 0.9)
    for ep in range(args.epochs):
        tl, tm = 0.0, 0
        for stp, (x, y) in enumerate(fake_loader(args.batch, args.iters, dev)):
            loss = res(net(x), y)
            loss.backward()
            opt.step()
            net.zero_grad(set_to_none=True)
            tl += loss.item() * x.size(0)
            tm += x.size(0)
            if stp % 10 == 0:
                log(f"sg ep{ep} stp{stp}", 0)
        print(f"ep{ep} loss={tl/tm:.3f}")


# ---------- Pipeline 主程 ----------
def pipeline(args):
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    rpc.init_rpc(
        f"w{rank}",
        rank=rank,
        world_size=world,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=128),
    )
    log("rpc ok", rank)

    if rank == 0:
        drv = Driver([f"w{i}" for i in range(world)], args.chunks)
        for ep in range(args.epochs):
            loss = drv.train_epoch(fake_loader(args.batch, args.iters, "cuda:0"), rank)
            log(f"epoch{ep} done loss={loss:.3f}", rank)
    rpc.shutdown()


# ---------- CLI ----------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--mode", choices=["single", "pipeline"], default="pipeline")
    pa.add_argument("--epochs", type=int, default=2)
    pa.add_argument("--batch", type=int, default=128)
    pa.add_argument("--chunks", type=int, default=4)
    pa.add_argument("--iters", type=int, default=40)
    cfg = pa.parse_args()
    if cfg.mode == "single":
        single(cfg)
    else:
        pipeline(cfg)
