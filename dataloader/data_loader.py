import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from pathlib import Path
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from collections import defaultdict
from .data_transforms import *
from RandAugment import RandAugment
from clip_model import clip
from scipy import sparse


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def video_no(self):
        return self._data[0]

    @property
    def video_id(self):
        return int(self._data[0])-1

    @property
    def label(self):
        return self._data[1]


class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class Kinetics_DataLoader(Dataset):
    def __init__(
            self,
            list_file,
            frame_path="dataset/frames/",
            graph_path="dataset/kg/",
            max_frames=8,
            n_px=224,
            isTraining=False
    ):
        self.list_file = list_file
        self.frame_path = frame_path
        self.graph_path = graph_path
        self.max_frames = max_frames
        self.n_px = n_px
        self.isTraining = isTraining

        self.transform = self._transform()

        if self.isTraining:
            self.transform.transforms.insert(0, GroupTransform(RandAugment(2, 9)))

        self._parse_list()
        self._kg_load()

    @property
    def categories(self):
        return [[i, l] for i, l in self.event_idx.items()]

    def _normalize_adj(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1.0).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sparse.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def _kg_load(self):
        with open(os.path.join(self.graph_path, 'event_idx.json'), 'r') as fp:
            self.event_idx = json.load(fp)
        with open(os.path.join(self.graph_path, 'subevent_idx.json'), 'r') as fp:
            self.subevent_idx = json.load(fp)
        self.event_idx = {int(k):v for k,v in self.event_idx.items()}
        self.subevent_idx = {int(k):v for k,v in self.subevent_idx.items()}
        self.event_idx_inv = {v:k for k,v in self.event_idx.items()}
        self.subevents = clip.tokenize(list(self.subevent_idx.values()))  # (459, 77)
        text_aug = f"Action of {{}}"
        # text_aug = [f"Human action of {{}}", f"{{}}, an action",
        #             f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
        #             f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
        #             f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
        #             f"The man is {{}}", f"The woman is {{}}"]
        self.events = clip.tokenize([text_aug.format(d) for d in self.event_idx_inv]) # (24, 77)

        Ascr = sparse.load_npz(os.path.join(self.graph_path, 'cat_des.npz'))
        A_shape = Ascr.shape
        Acoo = self._normalize_adj(Ascr).tocoo()
        self.cat_des = torch.sparse_coo_tensor([Acoo.row.tolist(), Acoo.col.tolist()],
                              torch.FloatTensor(Acoo.data.astype(np.float32)), A_shape) # (24, 459)
        A_shape = Ascr.T.shape
        Acoo = self._normalize_adj(Ascr.T).tocoo()
        self.des_cat = torch.sparse_coo_tensor([Acoo.row.tolist(), Acoo.col.tolist()],
                              torch.FloatTensor(Acoo.data.astype(np.float32)), A_shape) # ( 459, 24)

        Ascr = sparse.load_npz(os.path.join(self.graph_path, 'video_des.npz'))
        A_shape = Ascr.shape
        Acoo = self._normalize_adj(Ascr).tocoo()
        self.video_des = torch.sparse_coo_tensor([Acoo.row.tolist(), Acoo.col.tolist()],  torch.FloatTensor(Acoo.data.astype(np.float32)), A_shape)
        A_shape = Ascr.T.shape
        Acoo = self._normalize_adj(Ascr.T).tocoo()
        self.des_video = torch.sparse_coo_tensor([Acoo.row.tolist(), Acoo.col.tolist()],  torch.FloatTensor(Acoo.data.astype(np.float32)), A_shape)

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split('\t')) for x in open(self.list_file)]

    def _transform(self):
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        scale_size = self.n_px * 256 // 224
        if self.isTraining:

            unique = torchvision.transforms.Compose([GroupMultiScaleCrop(self.n_px, [1, .875, .75, .66]),
                                                     GroupRandomHorizontalFlip(is_sth=False),
                                                     GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4,
                                                                            saturation=0.2, hue=0.1),
                                                     GroupRandomGrayscale(p=0.2),
                                                     GroupGaussianBlur(p=0.0),
                                                     GroupSolarization(p=0.0)]
                                                    )
        else:
            unique = torchvision.transforms.Compose([GroupScale(scale_size),
                                                     GroupCenterCrop(self.n_px)])

        common = torchvision.transforms.Compose([Stack(roll=False),
                                                 ToTorchFormatTensor(div=True),
                                                 GroupNormalize(input_mean,
                                                                input_std)])
        return torchvision.transforms.Compose([unique, common])

    def _sample_indices(self, num_frames):
        if num_frames <= self.max_frames:
            offsets = np.concatenate((
                np.arange(num_frames),
                np.random.randint(num_frames,
                        size=self.max_frames - num_frames)))
            return np.sort(offsets)
        offsets = list()
        ticks = [i * num_frames // self.max_frames for i in range(self.max_frames + 1)]

        for i in range(self.max_frames):
            tick_len = ticks[i + 1] - ticks[i]
            tick = ticks[i]
            if tick_len >= 1:
                tick += np.random.randint(tick_len)
            offsets.extend([j for j in range(tick, tick + 1)])
        return np.array(offsets)

    def _get_val_indices(self, num_frames):
        if self.max_frames == 1:
            return np.array([num_frames //2], dtype=np.int_)

        if num_frames <= self.max_frames:
            return np.array([i * num_frames // self.max_frames
                             for i in range(self.max_frames)], dtype=np.int_)
        offset = (num_frames / self.max_frames - 1) / 2.0
        return np.array([i * num_frames / self.max_frames + offset
                         for i in range(self.max_frames)], dtype=np.int_)

    def _load_image(self, filepath):
        return [Image.open(filepath).convert('RGB')]

    def _load_knowledge(self, record):
        vno = record.video_no
        vid = record.video_id
        img_dir = os.path.join(self.frame_path, vno)

        filenames = [i for i in os.listdir(img_dir) if i.endswith('.jpg') & i.startswith('img')]
        filenames.sort(key=lambda x:int(x.split('.')[0].split('_')[1]))
        nb_frame = len(filenames)
        try:
            segment_indices = self._sample_indices(nb_frame) if self.isTraining else self._get_val_indices(nb_frame)
        except ValueError:
            print(vno)
        filenames = [filenames[i] for i in segment_indices]

        images = []
        for i, filename in enumerate(filenames):
            try:
                image = self._load_image(os.path.join(img_dir,filename))
                images.extend(image)
            except OSError:
                print('ERROR: Could not load the image!')
                raise

        frame_embs = self.transform(images)
        label_id = self.event_idx_inv[record.label]
        return frame_embs, self.video_des.to_dense()[vid], self.des_video.to_dense()[:,vid], label_id

    def __getitem__(self, idx):
        record = self.video_list[idx]
        return self._load_knowledge(record)

    def __len__(self):
        return len(self.video_list)
