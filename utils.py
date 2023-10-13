import torch
import torch.nn as nn
from torch._utils import ExceptionWrapper
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR
import os, io, random, threading, time
from pathlib import Path
import numpy as np
from collections import defaultdict, deque, OrderedDict
from copy import deepcopy
import datetime


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = -1
        world_size = -1
        local_rank = -1

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, timeout=timedelta(seconds=30))
    torch.distributed.barrier(device_ids=[local_rank])

    seed = args.seed + dist.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    return rank, world_size, local_rank, device, batch_size

def set_default_logger(args):
    global logger
    # pre-defining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    return args

def save_model(args, cmodel, model, optimizer, epoch):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    name = os.path.join(args.output_dir, "checkpoint-%s.pt" % epoch_name)
    torch.save({
        'epoch': epoch,
        'clip_model_state_dict': cmodel.state_dict(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, name)

def create_optimizer(args, cmodel, model):
    optimizer = optim.AdamW([{'params': cmodel.parameters()},
                             {'params': model.parameters(), 'lr': args.lr * 10}],
                            betas=(0.9, 0.98), lr=args.lr, eps=1e-8,
                            weight_decay=args.weight_decay)
    return optimizer

def lr_scheduler(args, optimizer):
    lr_scheduler = WarmupCosineAnnealingLR(
        optimizer,
        args.epochs,
        warmup_epochs=args.lr_warmup_step
    )
    return lr_scheduler

def generate_label(labels):
    num = len(labels)
    # gt = np.zeros(shape=(num, num))
    # for i, label in enumerate(labels):
    #     for k in range(num):
    #         if labels[k] == label:
    #             gt[i,k] = 1
    gt = np.zeros(shape=(num, 24))
    for i, label in enumerate(labels):
        gt[i, label] = 1
    return gt

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()
