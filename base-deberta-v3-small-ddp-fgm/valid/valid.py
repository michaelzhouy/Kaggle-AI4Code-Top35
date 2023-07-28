import numpy as np
import pandas as pd
import random
import json
from bisect import bisect
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions


def kendall_tau(ground_truth, predictions):
    total_inversions = 0
    total_2max = 0  # twice the maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max


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


def main():
    seed = 601
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # param
    sample = True
    debug = False
    model_name_or_path = 'microsoft/deberta-v3-small'
    valid_batch_size = 100
    md_max_len = 64
    code_max_len = 25
    total_max_len = 1024
    ckpt_paths = [f'../train/output/model_{i}.bin' for i in range(4, 5)]

    # path
    data_dir = Path('../../input/AI4Code')
    if sample:
        valid_mark_path = '../../input/data/valid_mark_sample.feather'
        valid_features_path = '../../input/data/valid_fts_sample.json'
        valid_path = '../../input/data/valid_sample.feather'
    else:
        valid_mark_path = '../../input/data/valid_mark.feather'
        valid_features_path = '../../input/data/valid_fts.json'
        valid_path = '../../input/data/valid.feather'
    
    # prepare input
    valid_df_mark = pd.read_feather(valid_mark_path).drop('parent_id', axis=1).dropna().reset_index(drop=True)
    valid_fts = json.load(open(valid_features_path))
    valid_df = pd.read_feather(valid_path)
    df_orders = pd.read_csv(data_dir / 'train_orders.csv', index_col='id').squeeze('columns').str.split()

    if debug:
        valid_df_mark = valid_df_mark[valid_df_mark['id'].isin(valid_df_mark['id'].unique()[:50])]
        valid_df = valid_df[valid_df['id'].isin(valid_df_mark['id'].unique()[:50])]

    # build model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    valid_ds = MarkdownDataset(valid_df_mark, tokenizer=tokenizer, md_max_len=md_max_len,
                               total_max_len=total_max_len, code_max_len=code_max_len, fts=valid_fts)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=valid_batch_size, shuffle=False, num_workers=8,
                              pin_memory=False, drop_last=False)

    
    for i, ckpt_path in enumerate(ckpt_paths):
        model = MarkdownModel(model_name_or_path)
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))

        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        preds = []
        labels = []
        model.eval()
        with torch.no_grad():
            for j, data in enumerate(tqdm(valid_loader)):
                inputs = tuple(d.cuda() for d in data[:-1])
                target = data[-1].cuda()
                with torch.cuda.amp.autocast():
                    pred = model(*inputs)
                preds.append(pred.detach().cpu().numpy().ravel())
                labels.append(target.detach().cpu().numpy().ravel())

        y_val, y_pred = np.concatenate(labels), np.concatenate(preds)

        valid_df['pred'] = valid_df.groupby(['id', 'cell_type'])['rank'].rank(pct=True)
        valid_df.loc[valid_df['cell_type'] == 'markdown', 'pred'] = y_pred
        y_dummy = valid_df.sort_values('pred').groupby('id')['cell_id'].apply(list)
        print('epoch {} Preds score {}'.format(ckpt_path.split('_')[-1].split('.')[0], kendall_tau(df_orders.loc[y_dummy.index], y_dummy)))


if __name__ == "__main__":
    main()
