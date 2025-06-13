import os, time, argparse, threading, queue
from typing import List
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
import torch.distributed.optim as dist_optim
from torchvision.models import resnet18

START = time.time()


def log(m, r):
    print(f"[{time.time()-START:6.2f}s] rk{r}: {m}", flush=True)


# ---------------- 远程 Stage ----------------
class Stage(nn.Module):
    def __init__(self, sub):
        super().__init__()
        self.dev = torch.cuda.current_device()
        self.sub = sub.to(self.dev)

    def forward(self, x):  # x 是 CPU tensor
        y = self.sub(x.to(self.dev, non_blocking=True))
        return y.cpu()

    def params(self):
        return [rpc.RRef(p) for p in self.sub.parameters()]


# ---------------- Pipeline Driver ----------------
class Driver:
    def __init__(self, workers: List[str], micro: int):
        s0, s1 = split_resnet18()
        self.stg0 = rpc.remote(workers[0], Stage, (s0,))
        self.stg1 = rpc.remote(workers[1], Stage, (s1,))
        self.micro = micro
        self.crit = nn.CrossEntropyLoss()
        params = self.stg0.rpc_sync().params() + self.stg1.rpc_sync().params()
        self.opt = dist_optim.DistributedOptimizer(
            torch.optim.SGD, params, lr=0.05, momentum=0.9
        )

    def _push_batch(self, xs, ys):
        """
        xs, ys: lists of micro-batches (CPU tensors)
        """
        q_out = queue.Queue()

        # --- 1. 先把所有 micro 送入 stage-0 ---
        futs0 = []
        for mb in xs:
            futs0.append(self.stg0.rpc_async().forward(mb))

        # --- 2. 主线程轮询 stage-0 的完成事件，完成一个就送 stage-1 ---
        outs = []
        done = threading.Event()

        def feeder():
            for f0 in futs0:
                t = f0.wait()  # tensor from stage-0
                f1 = self.stg1.rpc_async().forward(t)  # async 发 stage-1
                outs.append(f1)
            done.set()

        threading.Thread(target=feeder, daemon=True).start()

        done.wait()  # 等所有都喂给 stage-1
        outs = [f.wait() for f in outs]  # 取回 stage-1 输出 (CPU Tensor)
        return outs  # list[Tensor]

    def train_epoch(self, loader, rank):
        totl, tot = 0.0, 0
        for step, (x, y) in enumerate(loader):
            xs, ys = x.chunk(self.micro), y.chunk(self.micro)
            with dist_autograd.context() as ctx:
                outs = self._push_batch(list(xs), list(ys))
                outs = [o.cuda() for o in outs]
                tgts = [t.cuda() for t in ys]
                loss = torch.stack([self.crit(o, t) for o, t in zip(outs, tgts)]).mean()
                dist_autograd.backward(ctx, [loss])
                self.opt.step(ctx)
            totl += loss.item() * x.size(0)
            tot += x.size(0)
            if step % 10 == 0:
                log(f"step{step} loss={totl/tot:.3f}", rank)
        return totl / tot


# ---------------- 其它工具 ----------------
def split_resnet18():
    m = resnet18()
    return [
        nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2),
        nn.Sequential(m.layer3, m.layer4, m.avgpool, nn.Flatten(1), m.fc),
    ]


def fake_loader(bsz, steps, dev):
    for _ in range(steps):
        yield torch.randn(bsz, 3, 224, 224, device=dev), torch.randint(
            0, 1000, (bsz,), device=dev
        )


# ---------------- main ----------------
def pipeline(args):
    rk = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rk % torch.cuda.device_count())
    rpc.init_rpc(
        f"w{rk}",
        rank=rk,
        world_size=world,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=128),
    )
    log("rpc ready", rk)

    if rk == 0:
        drv = Driver([f"w0", f"w1"], args.micro)
        for ep in range(args.epochs):
            loss = drv.train_epoch(fake_loader(args.batch, args.steps, "cuda:0"), rk)
            log(f"epoch{ep} done loss={loss:.3f}", rk)
    rpc.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--micro", type=int, default=4)
    cfg = ap.parse_args()
    pipeline(cfg)
