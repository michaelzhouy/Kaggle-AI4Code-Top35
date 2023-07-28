import numpy as np
import pandas as pd
import time
import random
import argparse
import json
import os
import sys
from bisect import bisect
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


class MarkdownModel(nn.Module):
    def __init__(self, model_path):
        super(MarkdownModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.fc = nn.Linear(768, 1)
        self.dropout1 = nn.Dropout(0.1)

    def forward(self, ids, mask, fts):
        x = self.model(ids, mask)[0]
        # x = torch.cat((x[:, 0, :], fts), 1)
        x = self.dropout1(x)
        x = self.fc(x[:, 0, :])
        return x


class MarkdownDataset(Dataset):
    def __init__(self, df, tokenizer, total_max_len, md_max_len, code_max_len, fts):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.total_max_len = total_max_len
        self.md_max_len = md_max_len
        self.code_max_len = code_max_len
        self.fts = fts

    def __getitem__(self, index):
        row = self.df.iloc[index]

        inputs = self.tokenizer.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            max_length=self.md_max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        code_inputs = self.tokenizer.batch_encode_plus(
            [str(x) for x in self.fts[row.id]['codes']],
            add_special_tokens=True,
            max_length=self.code_max_len,
            padding='max_length',
            truncation=True
        )
        n_md = self.fts[row.id]['total_md']
        n_code = self.fts[row.id]['total_md']
        if n_md + n_code == 0:
            fts = torch.FloatTensor([0])
        else:
            fts = torch.FloatTensor([n_md / (n_md + n_code)])

        ids = inputs['input_ids']
        for x in code_inputs['input_ids']:
            ids.extend(x[1:])
        ids = ids[:self.total_max_len]
        if len(ids) != self.total_max_len:
            ids = ids + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(ids))
        ids = torch.LongTensor(ids)

        mask = inputs['attention_mask']
        for x in code_inputs['attention_mask']:
            mask.extend(x[1:])
        mask = mask[:self.total_max_len]
        if len(mask) != self.total_max_len:
            mask = mask + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(mask))
        mask = torch.LongTensor(mask)

        assert len(ids) == self.total_max_len

        return ids, mask, fts, torch.FloatTensor([row.pct_rank])

    def __len__(self):
        return self.df.shape[0]


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}
    
    def attack(self, epsilon=0.1, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
    
    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--local_rank', type=int, default=-1, help='local_rank for distributed training on gpus')

    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.device = device

    seed = 601
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # param
    sample = 'all'
    debug = False
    model_name_or_path = 'microsoft/deberta-v3-small'
    lr = 5e-5
    train_batch_size = 18
    md_max_len = 64
    code_max_len = 25
    total_max_len = 1024
    accumulation_steps = 2
    epochs = 5

    # path
    if sample == 'all':
        train_mark_path = '../../input/data/df_mark.feather'
        train_features_path = '../../input/data/df_fts_code40.json'
    elif sample == 'sample':
        train_mark_path = '../../input/data/train_mark_sample.feather'
        train_features_path = '../../input/data/train_fts_sample.json'
    elif sample == 'debug':
        train_mark_path = '../../input/data/train_mark.feather'
        train_features_path = '../../input/data/train_fts.json'

    # prepare input
    train_df_mark = pd.read_feather(train_mark_path).drop('parent_id', axis=1).dropna().reset_index(drop=True)
    train_fts = json.load(open(train_features_path))

    if debug:
        epochs = 3
        train_df_mark = train_df_mark[train_df_mark['id'].isin(train_df_mark['id'].unique()[:50])]

    # build model
    if args.local_rank != 0:
        torch.distributed.barrier()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = MarkdownModel(model_name_or_path)
    # ckpt_path = './output/model_3.bin'
    # model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank)

    train_ds = MarkdownDataset(train_df_mark, tokenizer=tokenizer, md_max_len=md_max_len,
                               total_max_len=total_max_len, code_max_len=code_max_len, fts=train_fts)
    train_sampler = DistributedSampler(train_ds)
    train_loader = DataLoader(dataset=train_ds, sampler=train_sampler, batch_size=train_batch_size,
                              num_workers=8, pin_memory=False, drop_last=True)

    num_train_optimization_steps = int(epochs * len(train_loader) / accumulation_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)

    if args.local_rank == 0:
        start_time = time.time()

    scaler = GradScaler()
    criterion = torch.nn.L1Loss()
    fgm = FGM(model)
    for ep in range(epochs):
        losses = AverageMeter()
        model.train()
        for idx, data in enumerate(tqdm(train_loader)):
            inputs = tuple(d.cuda() for d in data[:-1])
            target = data[-1].cuda()
            with autocast():
                logits = model(*inputs)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()

            fgm.attack()
            with autocast():
                logits_adv = model(*inputs)
                loss_adv = criterion(logits_adv, target)
            scaler.scale(loss_adv).backward()
            fgm.restore()
            
            losses.update(loss.item(), logits.size(0))
            
            if idx % accumulation_steps == 0 or idx == len(train_loader) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        if args.local_rank == 0:
            print('epoch: {}, train_loss: {}'.format(ep, losses.avg), flush=True)

            out_dir = './output/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            torch.save(model.module.state_dict(), out_dir + f'model_{ep}.bin')

            end_time = time.time()
            print('Consume: ', end_time - start_time)


if __name__ == "__main__":
    main()
