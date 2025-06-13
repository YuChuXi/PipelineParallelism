import os
import time
import logging
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import ResNet, BasicBlock
import matplotlib.pyplot as plt
import numpy as np

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# 自定义数据集类
class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=(3, 224, 224), num_classes=10):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = torch.randn(*self.image_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label

# 拆分ResNet模型
class PipelineResNet(ResNet):
    def __init__(self, stages, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stages = nn.ModuleList(stages)
        
    def forward(self, x, stage_idx):
        """执行指定阶段的前向传播"""
        logger.debug(f"Stage {stage_idx} forward start")
        start_time = time.time()
        
        if stage_idx == 0:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        elif stage_idx == len(self.stages) - 1:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        else:
            x = self.stages[stage_idx - 1](x)
        
        elapsed = time.time() - start_time
        logger.debug(f"Stage {stage_idx} forward end ({elapsed:.4f}s)")
        return x

# 流水线并行训练器
class PipelineTrainer:
    def __init__(self, model, device_ids, num_microbatches=8):
        self.model = model
        self.device_ids = device_ids
        self.num_stages = len(device_ids)
        self.num_microbatches = num_microbatches
        self.stage_times = [[] for _ in range(self.num_stages)]
        self.comm_times = []
        
        # 将每个阶段分配到不同设备
        self.stage_devices = [torch.device(f'cuda:{gpu_id}') for gpu_id in device_ids]
        for i, device in enumerate(self.stage_devices):
            self.model.stages[i] = self.model.stages[i].to(device)
            
    def train_step(self, data, target):
        # 将batch拆分为微批次
        microbatch_size = data.size(0) // self.num_microbatches
        losses = []
        
        # 流水线执行
        for mb_idx in range(self.num_microbatches + self.num_stages - 1):
            stage_idx = mb_idx % self.num_stages
            real_mb_idx = mb_idx - stage_idx
            
            # 执行前向传播
            if 0 <= real_mb_idx < self.num_microbatches:
                # 准备数据
                start_idx = real_mb_idx * microbatch_size
                end_idx = start_idx + microbatch_size
                mb_data = data[start_idx:end_idx]
                mb_target = target[start_idx:end_idx]
                
                # 移动到当前阶段设备
                if stage_idx == 0:
                    current_input = mb_data.to(self.stage_devices[0])
                else:
                    # 从前一阶段获取输入
                    comm_start = time.time()
                    current_input = current_output.to(self.stage_devices[stage_idx])
                    comm_time = time.time() - comm_start
                    self.comm_times.append(comm_time)
                    logger.debug(f"Stage {stage_idx} comm time: {comm_time:.4f}s")
                
                # 执行当前阶段计算
                stage_start = time.time()
                current_output = self.model(current_input, stage_idx)
                stage_time = time.time() - stage_start
                self.stage_times[stage_idx].append(stage_time)
                
                # 最后一个阶段计算损失
                if stage_idx == self.num_stages - 1:
                    loss = nn.functional.cross_entropy(current_output, mb_target.to(self.stage_devices[-1]))
                    losses.append(loss)
        
        # 聚合损失并执行反向传播
        if losses:
            total_loss = sum(losses) / len(losses)
            total_loss.backward()
            return total_loss.item()
        return 0.0

# 单机单卡训练器
class SingleGPUTrainer:
    def __init__(self, model, device_id=0):
        self.device = torch.device(f'cuda:{device_id}')
        self.model = model.to(self.device)
        self.times = []
        
    def train_step(self, data, target):
        start_time = time.time()
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data, 0)  # stage_idx=0 表示完整模型
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        step_time = time.time() - start_time
        self.times.append(step_time)
        return loss.item()

# 创建ResNet模型
def create_resnet(pretrained=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    # 拆分模型为多个阶段
    stages = nn.ModuleList([
        nn.Sequential(model.layer1),
        nn.Sequential(model.layer2),
        nn.Sequential(model.layer3),
        nn.Sequential(model.layer4)
    ])
    return PipelineResNet(stages, BasicBlock, [2, 2, 2, 2])

# 训练函数
def train(rank, world_size, args):
    # 初始化分布式环境
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
    # 创建模型和数据
    torch.cuda.set_device(rank)
    model = create_resnet()
    dataset = SyntheticDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    
    # 创建不同的训练器
    if args.mode == 'pipeline':
        device_ids = list(range(world_size))
        trainer = PipelineTrainer(model, device_ids, num_microbatches=8)
    else:  # single-gpu
        trainer = SingleGPUTrainer(model, rank)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # 训练循环
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for i, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            loss = trainer.train_step(data, target)
            optimizer.step()
            
            if i % 10 == 0:
                logger.info(f"Rank {rank} Epoch {epoch} Step {i} Loss: {loss:.4f}")
    
    # 收集时间统计信息
    if args.mode == 'pipeline' and rank == 0:
        return {
            'stage_times': trainer.stage_times,
            'comm_times': trainer.comm_times
        }
    elif args.mode == 'single':
        return {'step_times': trainer.times}
    return {}

# 可视化结果
def visualize_results(pipeline_results, single_results):
    plt.figure(figsize=(15, 10))
    
    # 流水线各阶段计算时间
    plt.subplot(2, 2, 1)
    for i, times in enumerate(pipeline_results['stage_times']):
        plt.plot(times, label=f'Stage {i}')
    plt.title('Pipeline Stage Computation Times')
    plt.xlabel('Microbatch Index')
    plt.ylabel('Time (s)')
    plt.legend()
    
    # 通信时间分布
    plt.subplot(2, 2, 2)
    plt.hist(pipeline_results['comm_times'], bins=50)
    plt.title('Communication Time Distribution')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')
    
    # 单卡训练时间
    plt.subplot(2, 2, 3)
    plt.plot(single_results['step_times'])
    plt.title('Single GPU Step Times')
    plt.xlabel('Step')
    plt.ylabel('Time (s)')
    
    # 效率对比
    plt.subplot(2, 2, 4)
    avg_pipeline_time = np.mean([t for stage in pipeline_results['stage_times'] for t in stage])
    avg_single_time = np.mean(single_results['step_times'])
    comm_percentage = 100 * np.mean(pipeline_results['comm_times']) / avg_pipeline_time
    
    labels = ['Single GPU', f'Pipeline ({len(pipeline_results["stage_times"])} stages)']
    times = [avg_single_time, avg_pipeline_time]
    
    plt.bar(labels, times, color=['blue', 'orange'])
    plt.title('Average Step Time Comparison')
    plt.ylabel('Time (s)')
    plt.text(1, avg_pipeline_time/2, f'Comm: {comm_percentage:.1f}%', ha='center', color='white')
    
    plt.tight_layout()
    plt.savefig('pipeline_vs_single.png')
    plt.show()

# 主函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pipeline', 'single'], default='pipeline',
                       help='Training mode: pipeline or single GPU')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Input batch size')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs to train')
    parser.add_argument('--master-addr', default='localhost',
                       help='Master address for distributed training')
    parser.add_argument('--master-port', default='29500',
                       help='Master port for distributed training')
    parser.add_argument('--nodes', type=int, default=2,
                       help='Number of nodes')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs per node')
    args = parser.parse_args()

    world_size = args.nodes * args.gpus
    
    # 运行分布式训练
    if args.mode == 'pipeline':
        mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
    else:
        # 单机单卡模式
        results = train(0, 1, args)
        print(f"Single GPU average step time: {np.mean(results['step_times']):.4f}s")

# 运行对比实验
def run_comparison():
    # 运行流水线并行
    print("Running pipeline training...")
    pipeline_args = argparse.Namespace(
        mode='pipeline',
        batch_size=32,
        epochs=2,
        master_addr='localhost',
        master_port='29500',
        nodes=1,
        gpus=4
    )
    pipeline_results = train(0, 4, pipeline_args)  # 假设在rank0收集结果
    
    # 运行单机单卡
    print("Running single GPU training...")
    single_args = argparse.Namespace(
        mode='single',
        batch_size=32,
        epochs=2,
        master_addr='localhost',
        master_port='29500',
        nodes=1,
        gpus=1
    )
    try:
        single_results = train(0, 1, single_args)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print("Single GPU OOM detected! Using smaller batch size")
            single_args.batch_size = 16
            single_results = train(0, 1, single_args)
        else:
            raise
    
    # 可视化结果
    visualize_results(pipeline_results, single_results)

if __name__ == '__main__':
    main()