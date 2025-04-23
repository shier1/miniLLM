import os
import io
import torch
import argparse
import torch.nn as nn
import torch.distributed as dist

from torchsummary import summary
from contextlib import redirect_stdout, nullcontext
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler

from model.model import get_model
from utils.log import Logger
from utils.config import get_config
from utils.dataset import PretrainedDataset
from utils.lr_scheduler import get_lr_scheduler
from utils.optimizer import get_optimizer


def get_argument():
    paser = argparse.ArgumentParser()
    paser.add_argument("--config_path", type=str, default="./configs/miniLLM.yaml")
    args = paser.parse_args()
    return args

def init_dist_mode(ddp):
    if not ddp: return
    global ddp_local_rank, DEVICE
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])              #RANK: 进程的序号，一般一个GPU对应一个全局的序号
    ddp_local_rank = int(os.environ["LOCAL_RANK"])  #LOCAL_RANK: 每一个进程在主机中的序号
    ddp_world_size = int(os.environ["WORD_SIZE"])   #WORD_SIZE: 获取当前启动的所有的进程的数量
    DEVICE = f"cuda:{ddp_local_rank}"               #当前机器上的cuda
    torch.cuda.set_device(DEVICE)

def train_one_epoch(model,
                    optimizer,
                    scheduler,
                    scaler,
                    ctx,
                    loss_fct,
                    logger):
    ...

if __name__ == "__main__":
    args = get_argument()
    config = get_config(args.config_path)
    logger = Logger().get_logger(config)
    # 初始化模型
    model = get_model(config)

    # 打印模型信息
    logger.info(f"the model named: {config.model.name}")
    outputbuffer = io.StringIO()
    with redirect_stdout(outputbuffer):
        summary(model.to(torch.device('cuda:0')), input_size=(256,32,32))
    logger.info(f"the summary parameters is:{outputbuffer.getvalue()}")

    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    # 设置混合精度训练
    ctx = nullcontext if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1   # 判断是否存在ddp环境
    ddp_local_rank, DEVICE = 0, "cuda:0"

    # 固定随机数种子和ddp的随机数种子
    base_seed = 127
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_dist_mode(ddp)
        device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)
    
    # 训练准备
    train_dataset = PretrainedDataset(config)
    train_sampler = DistributedSampler(train_dataset) if ddp else None
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.train.batch_size,
                                  pin_memory=True,
                                  drop_last=False,
                                  shuffle=False,
                                  num_workers=config.train.num_workers,
                                  sampler=train_sampler)

    # 设置lr，optimizer， scaler等配置
    scaler = GradScaler(enabled=config.amp_scaler)
    optimizer = get_optimizer(config, model)
    scheduler = get_lr_scheduler(config)
    loss_fct = nn.CrossEntropyLoss(reduce="none")

    # 配置多卡
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 开始训练
    for epoch in range(config.epochs):
        train_one_epoch(model, 
                        optimizer,
                        scheduler,
                        scaler,
                        ctx,
                        loss_fct,
                        logger)
