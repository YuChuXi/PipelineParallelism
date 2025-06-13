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
START = time.time()


def log(msg, rank):
    print(f"[{time.time()-START:6.2f}s] rk{rank}: {msg}", flush=True)


# ---------- 随机数据 ----------
def synthetic_loader(batch: int, steps: int, dev: str):
    for _ in range(steps):
        yield torch.randn(batch, 3, 224, 224, device=dev), torch.randint(
            0, 1000, (batch,), device=dev
        )


# ---------- 模型切两段 ----------
def split_resnet18():
    m = resnet18(num_classes=1000)
    s0 = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2)
    s1 = nn.Sequential(m.layer3, m.layer4, m.avgpool, nn.Flatten(1), m.fc)
    return [s0, s1]


# ---------- 远程 Stage ----------
class Stage(nn.Module):
    def __init__(self, sub: nn.Module):
        super().__init__()
        self.dev = torch.cuda.current_device()
        self.sub = sub.to(self.dev)

    def forward(self, x: torch.Tensor):
        y = self.sub(x.to(self.dev, non_blocking=True))
        return y.cpu()  # 返回 CPU Tensor

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.sub.parameters()]


# ---------- Driver ----------
class PipelineDriver:
    def __init__(self, workers: List[str], chunks: int):
        self.chunks = chunks
        self.stage_rrefs = [
            rpc.remote(w, Stage, (s,)) for w, s in zip(workers, split_resnet18())
        ]
        self.param_rrefs = list(
            itertools.chain.from_iterable(
                s.rpc_sync().parameter_rrefs() for s in self.stage_rrefs
            )
        )
        self.crit = nn.CrossEntropyLoss()
        self.opt = dist_optim.DistributedOptimizer(
            torch.optim.SGD, self.param_rrefs, lr=0.05, momentum=0.9
        )

    # ---- 运行 1 个 micro-batch，返回 *Future[Tensor]* ----
    def _launch_one(self, mb: torch.Tensor):
        fut = self.stage_rrefs[0].rpc_async().forward(mb.cpu())
        for nxt in self.stage_rrefs[1:]:
            # flatten: wait 前一段结果，再异步发给下一段，返回 Future[Tensor]
            fut = fut.then(lambda t, r=nxt: r.rpc_async().forward(t).wait())
        return fut  # 最外层已是 tensor future

    def train_epoch(self, loader, rank):
        tot_loss, tot = 0.0, 0
        for step, (x, y) in enumerate(loader):
            micro_x, micro_y = x.chunk(self.chunks), y.chunk(self.chunks)
            with dist_autograd.context() as ctx:
                futs = [self._launch_one(mb) for mb in micro_x]
                outs = [f.wait().cuda() for f in futs]  # 都是 Tensor
                tgts = [t.cuda() for t in micro_y]
                loss = torch.stack([self.crit(o, t) for o, t in zip(outs, tgts)]).mean()
                dist_autograd.backward(ctx, [loss])
                self.opt.step(ctx)
            tot_loss += loss.item() * x.size(0)
            tot += x.size(0)
            if step % 10 == 0:
                log(f"step{step}  loss={tot_loss/tot:.3f}", rank)
        return tot_loss / tot


# ---------- 单机基准 ----------
def single_gpu(args):
    dev = "cuda:0"
    torch.cuda.set_device(0)
    net, res = resnet18().to(dev), nn.CrossEntropyLoss()
    opt = torch.optim.SGD(net.parameters(), 0.05, 0.9)
    tic = time.time()
    for ep in range(args.epochs):
        loader = synthetic_loader(args.batch, args.iters, dev)
        tl, tm = 0.0, 0
        for stp, (x, y) in enumerate(loader):
            loss = res(net(x), y)
            loss.backward()
            opt.step()
            net.zero_grad(set_to_none=True)
            tl += loss.item() * x.size(0)
            tm += x.size(0)
            if stp % 10 == 0:
                log(f"sg ep{ep} stp{stp}", 0)
        print(f"ep{ep} loss={tl/tm:.3f}")
    t = time.time() - tic
    print(f"single total {t:.2f}s")
    return t


# ---------- 可视化 ----------
def plot(t1, t2, g):
    sp = t1 / t2
    fig, ax = plt.subplots()
    ax.bar(["single"], [t1])
    ax.bar([f"{g}-gpu"], [t2])
    ax.set_title(f"speed-up {sp:.2f}×")
    ax.set_ylabel("sec")
    plt.tight_layout()
    plt.savefig("speed.png")
    print("fig saved: speed.png")


# ---------- Pipeline 主程 ----------
def run_pipe(args):
    rk = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rk % torch.cuda.device_count())
    rpc.init_rpc(
        f"w{rk}",
        rank=rk,
        world_size=ws,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=128),
    )
    log("rpc ok", rk)
    if rk == 0:
        drv = PipelineDriver([f"w{i}" for i in range(ws)], args.chunks)
        tic = time.time()
        for ep in range(args.epochs):
            loss = drv.train_epoch(
                synthetic_loader(args.batch, args.iters, "cuda:0"), rk
            )
            log(f"epoch{ep} done loss={loss:.3f}", rk)
        t = time.time() - tic
        log(f"pipe total {t:.2f}s", rk)
        if args.single_time:
            plot(args.single_time, t, ws)
    rpc.shutdown()


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "pipeline"], default="single")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--chunks", type=int, default=4)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--single_time", type=float, default=None)
    a = p.parse_args()
    if a.mode == "single":
        t = single_gpu(a)
        print("把该用时填入 --single_time 再跑 pipeline 可画对比图")
    else:
        run_pipe(a)


if __name__ == "__main__":
    main()
