import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
# import wandb
import argparse
import shutil
from pathlib import Path
import pprint
import random
import numpy as np
import utils
from clip_model import simple_tokenizer, clip
import clip_model
import warnings
from modules import *
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
from dataloader import Kinetics_DataLoader
from engine import train_epoch, evaluate
warnings.filterwarnings('ignore')


def print_only_rank0(log):
    if dist.get_rank() == 0:
        print(log)

def get_args(description='Kinetics_CLIP'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--network_type", default="Kinetics_CLIP", type=str, help="A name of this network")
    parser.add_argument('--log_time', default='')
    # data arguments
    parser.add_argument('--dataset', default='Kinetics')
    parser.add_argument('--train_list', type=str, default="./dataset/full-shot/82/train_list.txt", help='')
    parser.add_argument('--val_list', type=str, default="./dataset/full-shot/82/test_list.txt", help='')
    parser.add_argument('--test_list', type=str, default="./dataset/full-shot/82/test_list.txt", help='')

    parser.add_argument('--graph_path', type=str, default="./dataset/kg/", help='annotation file path')
    parser.add_argument('--frame_path', type=str, default="../data/frames/", help='frames file path')
    parser.add_argument('--max_frames', type=int, default=4, help='max frames of each video')

    # training arguments
    parser.add_argument("--isTraining", action='store_true', help="Whether to run training.")
    parser.add_argument('--num_workers', type=int, default=8, help='')
    parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--weight_decay', type=float, default=0.2, help='a regularization term to penalize big weights')
    parser.add_argument('--lr_warmup_step', type=int, default=5, help='')

    parser.add_argument('--seed', type=int, default=1024, help='random seed')
    parser.add_argument('--eval_freq', default=1, type=int)

    # output arguments
    parser.add_argument("--output_dir", default="./outputs/", type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir", default="../data/pretrained_models/", type=str,
                        help="Where do you want to store the pre-trained models of CLIP")
    parser.add_argument("--network_arch", default="ViT-B/32", type=str, help="Initial model.")

    # pre-trained model arguments
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "LSTM", "Transf", "Conv_1D"],
                        help="choice a similarity header.")
    known_args, _ = parser.parse_known_args()

    return parser.parse_args()


def main():
    args = get_args()
    args.device = "cuda:3" if torch.cuda.is_available() else "cpu"
    args = utils.set_default_logger(args)

    print('-' * 80)
    print(' ' * 10, "{}: working on dataset {} with {}".format(args.network_type, args.dataset, args.network_arch))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print('-' * 80)

    cmodel, preprocess = clip.load(args.network_arch, device=args.device, download_root=args.cache_dir)

    state_dict = cmodel.state_dict()
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64

    text_enc = TextCLIP(cmodel).to(args.device)
    img_enc = ImageCLIP(cmodel).to(args.device)
    model = KGLearner(args, embed_dim, transformer_heads).to(args.device)

    # text_enc = torch.nn.DataParallel(text_enc).to(args.device)
    # img_enc = torch.nn.DataParallel(img_enc).to(args.device)
    # model = torch.nn.DataParallel(model).to(args.device)

    optimizer = utils.create_optimizer(args, cmodel, model)
    lr_scheduler = utils.lr_scheduler(args, optimizer)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) + \
                    sum(p.numel() for p in cmodel.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    context_length = cmodel.state_dict()["positional_embedding"].shape[0]

    if args.isTraining:
        train_data = Kinetics_DataLoader(list_file=args.train_list,
                                         graph_path=args.graph_path,
                                         frame_path=args.frame_path,
                                         max_frames=args.max_frames,
                                         n_px=224,
                                         isTraining=args.isTraining)

        train_dataloader = DataLoader(train_data,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)

        val_data = Kinetics_DataLoader(list_file=args.val_list,
                                       graph_path=args.graph_path,
                                       frame_path=args.frame_path,
                                       max_frames=args.max_frames,
                                       n_px=224)

        val_dataloader = DataLoader(val_data,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False,
                                    pin_memory=False,
                                    drop_last=False)

        global_step = 0
        best_score_acc1 = 0.00001
        best_score_acc5 = 0.00001
        event = val_data.events.to(args.device)
        subevent = val_data.subevents.to(args.device)
        cd_adj = val_data.cat_des.to(args.device)
        dc_adj = val_data.des_cat.to(args.device)

        for epoch in range(args.epochs):
            train_epoch_loss = train_epoch(epoch, args, model, cmodel, text_enc, img_enc, event, subevent, cd_adj, dc_adj, train_dataloader, optimizer, lr_scheduler)

            if (epoch + 1) % args.eval_freq == 0:
                ## Run on val dataset for validation.
                val_acc1 = evaluate(args, model, cmodel, text_enc, img_enc, event, subevent, cd_adj, dc_adj, val_dataloader)

                if best_score_acc1 <= val_acc1:
                    best_score_acc1 = val_acc1
                    utils.save_model(args, cmodel, model, optimizer, epoch="best_acc1")
                print('Epoch: [{}/{}]: Training loss： {:.2f}, Top1: {:.3f}%/{:.3f}%'.format(epoch, args.epochs, train_epoch_loss, val_acc1, best_score_acc1))

        print('The best predictions: Top1: {:.2f}%'.format(best_score_acc1))

    else:
        if args.device == "cpu":
            cmodel.float()
        else:
            clip_model.model.convert_weights(cmodel)
        try:
            print(("=> loading checkpoint checkpoint-best"))
            checkpoint = torch.load(os.path.join(args.output_dir, "checkpoint-best_acc1.pt"))
            model.load_state_dict(checkpoint['model_state_dict'])
            cmodel.load_state_dict(checkpoint['clip_model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            del checkpoint

        except ModelNotFound:
            print("=> no checkpoint '{}' found ".format(os.path.join(args.output_dir, "checkpoint-best_acc1")))
            raise
        val_data = Kinetics_DataLoader(list_file=args.test_list,
                                       graph_path=args.graph_path,
                                       frame_path=args.frame_path,
                                       max_frames=args.max_frames,
                                       n_px=224)

        val_dataloader = DataLoader(val_data,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False,
                                    pin_memory=False,
                                    drop_last=False)

        event = val_data.events.to(args.device)
        subevent = val_data.subevents.to(args.device)
        cd_adj = val_data.cat_des.to(args.device)
        dc_adj = val_data.des_cat.to(args.device)
        val_acc1 = evaluate(args, model, cmodel, text_enc, img_enc, event, subevent, cd_adj, dc_adj, val_dataloader)
        print('The test predictions: Top1: {:.2f}%'.format(val_acc1))
    # wandb.finish()

if __name__ == '__main__':
    main()
