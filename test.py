# -*- coding: utf-8 -*-
import argparse
import os
import time
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


# --- 1. 模型定义与切分 ---
# 将ResNet-50模型切分为多个部分，以适应流水线并行的需求
# world_size: GPU总数
def get_resnet_split(world_size):
    """
    将 torchvision 的 ResNet-50 模型切分成 world_size 个部分。
    """
    model = resnet50(weights=None) # 不加载预训练权重以避免网络下载
    
    # 为了演示，我们手动将模型切分为几个nn.Sequential块
    # 在实际应用中，你可能需要更智能的切分策略来平衡各部分计算量
    splits = []
    if world_size == 1:
        return [model]
    if world_size == 2:
        splits.append(nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2
        ))
        splits.append(nn.Sequential(
            model.layer3, model.layer4, model.avgpool, nn.Flatten(), model.fc
        ))
    elif world_size == 4:
        splits.append(nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool, model.layer1
        ))
        splits.append(model.layer2)
        splits.append(model.layer3)
        splits.append(nn.Sequential(
            model.layer4, model.avgpool, nn.Flatten(), model.fc
        ))
    else:
        raise ValueError(f"不支持 world_size={world_size} 的自动切分，请为2或4，或手动实现切分逻辑。")

    print(f"模型被成功切分为 {len(splits)} 部分，用于 {world_size} 卡流水线。")
    return splits

# --- 2. 流水线并行模块 ---
# 这个类封装了流水线并行的核心逻辑
class PipeModule(nn.Module):
    def __init__(self, model_parts, device, micro_batches):
        super().__init__()
        self.parts = []
        # 将模型的每个部分注册为子模块并移动到指定设备
        for i, part in enumerate(model_parts):
            self.parts.append(part.to(device))
            self.add_module(f"part_{i}", self.parts[i])

        self.rank = rpc.get_worker_info().id
        self.world_size = len(model_parts)
        self.device = device
        self.micro_batches = micro_batches
        self.loss_fn = nn.CrossEntropyLoss()
        
        # 用于形象化日志输出
        self.start_time = time.time()
        self.log_prefix = f"[T=%.2f] [GPU {self.rank}]"

    def _log(self, msg):
        current_time = time.time() - self.start_time
        print(self.log_prefix % current_time + " " + msg)

    def forward(self, x_rref, y_rref):
        # 从RRef (Remote Reference)中获取真实的输入数据和标签
        # 只有rank 0会接收到真实的输入，其他rank接收的是中间激活
        if self.rank == 0:
            inputs = x_rref.to_local().to(self.device)
            labels = y_rref.to_local().to(self.device)
            # 将一个大batch切分为多个micro-batch
            micro_inputs = torch.chunk(inputs, self.micro_batches, dim=0)
        else:
            labels = y_rref.to_local().to(self.device)
            micro_labels = torch.chunk(labels, self.micro_batches, dim=0)

        # 流水线核心逻辑
        # 使用 RRef 来持有对下一阶段输入的引用
        remote_inputs_rrefs = []
        
        # 预热阶段 (Warm-up)
        # 填充流水线
        for i in range(self.world_size - 1 - self.rank):
            if self.rank == 0:
                x = micro_inputs[i]
            else:
                # 从上一级获取中间激活
                x_rref = remote_inputs_rrefs.pop(0) if remote_inputs_rrefs else x_rref
                self._log(f"开始 FWD (micro-batch {i}) - 接收数据")
                x = x_rref.to_local()
                self._log(f"完成 FWD (micro-batch {i}) - 接收数据")

            self._log(f"开始 FWD (micro-batch {i}) - 计算")
            out = self.parts[self.rank](x)
            self._log(f"完成 FWD (micro-batch {i}) - 计算")
            
            # 将中间结果通过RPC发送到下一个rank
            next_rank = (self.rank + 1) % self.world_size
            self._log(f"开始 FWD (micro-batch {i}) - 发送到 GPU {next_rank}")
            remote_out_rref = rpc.remote(f"worker{next_rank}", lambda x: x, args=(out,))
            
            if self.rank < self.world_size - 2:
                # 如果不是倒数第二级，将下一级的输入RRef传递给下一级
                rpc.rpc_async(f"worker{next_rank}", 
                              PipeModule.forward, 
                              args=(remote_out_rref, RRef(micro_labels[i]) if self.rank == 0 else y_rref))
            else: # 倒数第二级，直接将结果发给最后一级
                remote_inputs_rrefs.append(remote_out_rref)

        # 稳定阶段 (Steady-state)
        # 流水线已满，每个GPU同时处理不同的micro-batch
        total_loss = 0
        for i in range(self.micro_batches):
            if self.rank == 0:
                # 第一个GPU处理新的micro-batch
                x = micro_inputs[i]
            else:
                # 其他GPU接收上一级的输出
                x_rref = remote_inputs_rrefs.pop(0) if remote_inputs_rrefs else x_rref
                self._log(f"开始 FWD (micro-batch {i}) - 接收数据")
                x = x_rref.to_local()
                self._log(f"完成 FWD (micro-batch {i}) - 接收数据")
            
            self._log(f"开始 FWD (micro-batch {i}) - 计算")
            out = self.parts[self.rank](x)
            self._log(f"完成 FWD (micro-batch {i}) - 计算")

            if self.rank < self.world_size - 1:
                # 不是最后一个GPU，将激活传递给下一个GPU
                next_rank = self.rank + 1
                self._log(f"开始 FWD (micro-batch {i}) - 发送到 GPU {next_rank}")
                remote_out_rref = rpc.remote(f"worker{next_rank}", lambda x: x, args=(out,))
                remote_inputs_rrefs.append(remote_out_rref)
            else:
                # 最后一个GPU，计算损失
                self._log(f"开始 LOSS (micro-batch {i})")
                loss = self.loss_fn(out, micro_labels[i])
                total_loss += loss
                self._log(f"完成 LOSS (micro-batch {i}), Loss: {loss.item():.4f}")
        
        return total_loss

    def get_parameters_rrefs(self):
        """获取模型所有参数的Remote References，用于分布式优化器。"""
        param_rrefs = []
        for param in self.parameters():
            param_rrefs.append(RRef(param))
        return param_rrefs

# --- 3. 分布式工作节点 (Worker) 逻辑 ---
def run_worker(rank, world_size, master_addr, master_port, batch_size, num_batches, micro_batches):
    """
    每个RPC worker进程执行的函数。
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=300 # 5分钟超时
    )

    # 初始化RPC
    print(f"Rank {rank}: 正在初始化RPC...")
    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options
    )
    print(f"Rank {rank}: RPC初始化成功。")

    # 获取模型切片
    device = f'cuda:{rank}'
    model_parts = get_resnet_split(world_size)
    
    # 将模型封装到PipeModule中
    model = PipeModule(model_parts, device, micro_batches)
    
    # 为分布式优化器收集所有参数的RRef
    param_rrefs = []
    for r in range(world_size):
        # 从每个worker获取其模型部分的参数引用
        param_rrefs.extend(rpc.rpc_sync(f"worker{r}", model.get_parameters_rrefs))

    # 创建分布式优化器
    optimizer = DistributedOptimizer(
        optim.Adam,
        param_rrefs,
        lr=0.001
    )

    # --- 训练循环 ---
    start_train_time = time.time()
    
    for i in range(num_batches):
        # 在rank 0上生成模拟数据
        if rank == 0:
            print("-" * 30)
            print(f"开始 Batch {i+1}/{num_batches}")
            # 使用分布式autograd上下文
            with dist_autograd.context() as context_id:
                # 准备输入数据和标签
                inputs = torch.randn(batch_size, 3, 224, 224, device=device)
                labels = torch.randint(0, 1000, (batch_size,), device=device)

                # 将数据封装在RRef中，启动流水线
                x_rref = RRef(inputs)
                y_rref = RRef(labels)
                
                # 在最后一个rank上异步执行前向传播和loss计算
                loss_future = rpc.rpc_async(f"worker{world_size-1}", model, args=(x_rref, y_rref))
                
                # 获取最终loss
                loss = loss_future.wait()
                
                model._log(f"Batch {i+1} 总损失: {loss.item():.4f}")
                model._log(f"开始 BWD (整个batch)")
                # 执行分布式反向传播
                dist_autograd.backward(context_id, [loss])
                model._log(f"完成 BWD (整个batch)")
                
                # 执行优化器步骤
                model._log(f"开始 Optimizer Step")
                optimizer.step(context_id)
                model._log(f"完成 Optimizer Step")

    end_train_time = time.time()
    total_time = end_train_time - start_train_time
    
    # 仅在rank 0上打印总结
    if rank == 0:
        throughput = (batch_size * num_batches) / total_time
        print("\n" + "="*50)
        print("流水线并行训练完成")
        print("="*50)
        print(f"  总批次数: {num_batches}")
        print(f"  每批大小: {batch_size}")
        print(f"  总耗时: {total_time:.2f} 秒")
        print(f"  吞吐量: {throughput:.2f} samples/sec")
        
        # 收集所有GPU的峰值显存
        mem_results = []
        for r in range(world_size):
            peak_mem = rpc.rpc_sync(f"worker{r}", torch.cuda.max_memory_allocated, args=(f'cuda:{r}',))
            mem_results.append(peak_mem / 1e9) # GB
            print(f"  GPU {r} 峰值显存: {mem_results[r]:.3f} GB")
        
        # 将结果保存用于对比
        result = {
            "mode": "Pipeline",
            "world_size": world_size,
            "throughput": throughput,
            "total_time": total_time,
            "peak_memory_gb": mem_results,
            "status": "Success"
        }
        torch.save(result, "pipeline_result.pt")

    # 关闭RPC
    rpc.shutdown()

# --- 4. 单机单卡基线测试 ---
def run_single_gpu(batch_size, num_batches):
    """
    在单个GPU上运行完整的模型，作为性能基准。
    """
    print("="*50)
    print("开始单机单卡基线测试 (on cuda:0)")
    print("="*50)
    
    device = 'cuda:0'
    result = {
        "mode": "Single GPU",
        "world_size": 1,
        "throughput": 0,
        "total_time": float('inf'),
        "peak_memory_gb": [0],
        "status": "Failure"
    }

    try:
        # 加载完整模型
        model = get_resnet_split(world_size=1)[0].to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        # 重置显存统计
        torch.cuda.reset_peak_memory_stats(device)

        start_time = time.time()
        for i in range(num_batches):
            print(f"Batch {i+1}/{num_batches}")
            inputs = torch.randn(batch_size, 3, 224, 224, device=device)
            labels = torch.randint(0, 1000, (batch_size,), device=device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = (batch_size * num_batches) / total_time
        peak_memory = torch.cuda.max_memory_allocated(device) / 1e9 # GB

        print("\n" + "="*50)
        print("单机单卡训练完成")
        print("="*50)
        print(f"  总耗时: {total_time:.2f} 秒")
        print(f"  吞吐量: {throughput:.2f} samples/sec")
        print(f"  GPU 0 峰值显存: {peak_memory:.3f} GB")

        result.update({
            "throughput": throughput,
            "total_time": total_time,
            "peak_memory_gb": [peak_memory],
            "status": "Success"
        })

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n" + "!"*50)
            print("错误：CUDA Out of Memory! 单机单卡模式无法在当前显存下运行。")
            print("!"*50)
            result["status"] = "OOM"
        else:
            print(f"\n发生未知运行时错误: {e}")
            result["status"] = f"Error: {e}"
    
    finally:
        torch.save(result, "single_gpu_result.pt")


# --- 5. 主函数和对比可视化 ---
def main():
    parser = argparse.ArgumentParser(description="PyTorch流水线并行Demo")
    parser.add_argument("--mode", type=str, default="pipeline", choices=["pipeline", "single", "compare"],
                        help="运行模式: 'pipeline' (分布式), 'single' (单卡基准), 'compare' (比较结果)")
    parser.add_argument("--rank", type=int, default=0, help="当前节点的rank")
    parser.add_argument("--world_size", type=int, default=4, help="总节点/GPU数量")
    parser.add_argument("--master_addr", type=str, default="localhost", help="主节点地址")
    parser.add_argument("--master_port", type=int, default=29500, help="主节点端口")
    parser.add_argument("--batch_size", type=int, default=256, help="总batch size")
    parser.add_argument("--num_batches", type=int, default=10, help="训练的batch数量")
    parser.add_argument("--micro_batches", type=int, default=8, help="流水线中的micro-batch数量")
    
    args = parser.parse_args()

    if args.mode == "pipeline":
        if not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size:
            if args.rank == 0:
                print(f"错误：需要 {args.world_size} 个GPU，但只检测到 {torch.cuda.device_count()} 个。")
            return
        run_worker(args.rank, args.world_size, args.master_addr, args.master_port, 
                   args.batch_size, args.num_batches, args.micro_batches)
    elif args.mode == "single":
        if not torch.cuda.is_available():
            print("错误：此模式需要至少一个GPU。")
            return
        run_single_gpu(args.batch_size, args.num_batches)
    elif args.mode == "compare":
        # 加载两个模式的运行结果并打印对比表格
        try:
            pipeline_res = torch.load("pipeline_result.pt")
            single_res = torch.load("single_gpu_result.pt")
            
            print("\n" + "="*80)
            print(" " * 30 + "性能对比报告")
            print("="*80)
            print("| 配置 (模式 | GPU数量) | 状态    | 吞吐量 (samples/sec) | 总耗时 (s) | 峰值显存 (GB)            |")
            print("|--------------------------|---------|------------------------|--------------|--------------------------|")
            
            # 单卡结果
            status_s = single_res['status']
            tp_s = f"{single_res['throughput']:.2f}" if status_s == 'Success' else 'N/A'
            time_s = f"{single_res['total_time']:.2f}" if status_s == 'Success' else 'N/A'
            mem_s = f"GPU0: {single_res['peak_memory_gb'][0]:.3f}" if status_s == 'Success' else 'N/A'
            print(f"| Single GPU (1 卡)        | {status_s:<7} | {tp_s:<22} | {time_s:<12} | {mem_s:<24} |")

            # 流水线结果
            status_p = pipeline_res['status']
            tp_p = f"{pipeline_res['throughput']:.2f}" if status_p == 'Success' else 'N/A'
            time_p = f"{pipeline_res['total_time']:.2f}" if status_p == 'Success' else 'N/A'
            mem_list = [f"GPU{i}: {m:.3f}" for i, m in enumerate(pipeline_res['peak_memory_gb'])]
            mem_p = ', '.join(mem_list)
            print(f"| Pipeline ({pipeline_res['world_size']} 卡)       | {status_p:<7} | {tp_p:<22} | {time_p:<12} | {mem_p:<24} |")
            print("-" * 80)

        except FileNotFoundError:
            print("\n错误：找不到结果文件 'pipeline_result.pt' 或 'single_gpu_result.pt'。")
            print("请先分别以 'pipeline' 和 'single' 模式运行脚本。")
            print("例如: python your_script.py --mode single")
            print("然后(在多个终端中): python your_script.py --mode pipeline --rank ...")

if __name__ == "__main__":
    main()
