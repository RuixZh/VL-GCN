import torch
import os
import sys
import utils
import numpy as np
from typing import Iterable, Optional
from tqdm import tqdm
import time
import clip_model


def train_epoch(epoch, args, model, cmodel, text_enc, img_enc, event, subevent, cd_adj, dc_adj, data_loader, optimizer, lr_scheduler):
    model.train()
    text_enc.train()
    img_enc.train()
    total_loss = 0.0
    length = len(data_loader)

    # learning video embedding afterwards
    with tqdm(total=len(data_loader), desc="Epoch %s"%epoch) as pbar:

        for kkk, (frames, vd_adj, dv_adj, label_id) in enumerate(data_loader):
            if (kkk+1) == 1 or (kkk+1) % 10 == 0:
                lr_scheduler.step(epoch + kkk / length)
            optimizer.zero_grad()

            # learning kg embedding first
            subevent_emb = text_enc(subevent)
            event_emb = text_enc(event)

            frames = frames.view((-1, args.max_frames, 3) + frames.size()[-2:])
            bs, nbf, c, h, w = frames.size()
            frames = frames.to(args.device).view(-1, c, h, w)
            frame_emb = img_enc(frames)  # (bs*nb_frame, dim)
            dim = frame_emb.size()[-1]
            frame_emb = frame_emb.view(-1, nbf, dim)  # (bs, nb_frame, dim)
            logit_scale = cmodel.logit_scale.exp()
            ground_truth = torch.tensor(label_id, dtype=torch.int64, device=args.device)
            vd_adj = vd_adj.to(args.device)
            dv_adj = dv_adj.to(args.device)

            loss, _ = model(frame_emb, cd_adj, dc_adj, vd_adj, dv_adj, subevent_emb, event_emb, logit_scale, ground_truth=ground_truth)
            total_loss += loss
            loss.backward()
            if args.device == "cpu":
                optimizer.step()
            else:
                utils.convert_models_to_fp32(cmodel)
                optimizer.step()
                clip_model.model.convert_weights(cmodel)
            # time.sleep(0.05)
            pbar.set_postfix({"loss":"{0:1.3f}".format(loss)})

            pbar.update(1)

    return total_loss / (kkk+1)


@torch.no_grad()
def evaluate(args, model, cmodel, text_enc, img_enc, event, subevent, cd_adj, dc_adj, data_loader):
    model.eval()
    text_enc.eval()
    img_enc.eval()
    num = 0
    corr_1 = 0
    corr_5 = 0
    total_acc1 = 0.0
    total_acc5 = 0.0
    # learning kg embedding first
    subevent_emb = text_enc(subevent)
    event_emb = text_enc(event)

    with tqdm(total=len(data_loader), desc="Evaluate") as pbar:
        for kkk, (frames, vd_adj, dv_adj, label_id) in enumerate(data_loader):

            frames = frames.view((-1, args.max_frames, 3) + frames.size()[-2:])
            bs, nbf, c, h, w = frames.size()
            frames = frames.to(args.device).view(-1, c, h, w)
            frame_emb = cmodel.encode_image(frames)  # (bs*nb_frame, dim)
            dim = frame_emb.size()[-1]
            frame_emb = frame_emb.view(-1, nbf, dim)  # (bs, nb_frame, dim)
            vd_adj = vd_adj.to(args.device)
            dv_adj = dv_adj.to(args.device)

            logit_scale = cmodel.logit_scale.exp()
            ground_truth = torch.tensor(label_id, dtype=torch.int64, device=args.device)

            _, indices_1 = model(frame_emb, cd_adj, dc_adj, vd_adj, dv_adj, subevent_emb, event_emb, logit_scale, ground_truth=ground_truth)

            num += bs
            for i in range(bs):
                if indices_1[i] == label_id[i]:
                    corr_1 += 1

            pbar.update(1)

    top1 = float(corr_1) / num * 100
    return top1
