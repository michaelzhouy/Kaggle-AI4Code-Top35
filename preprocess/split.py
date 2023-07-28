import json
from pathlib import Path
import numpy as np
import pandas as pd
import re
from scipy import sparse
from tqdm import tqdm
import os
from sklearn.model_selection import GroupShuffleSplit


# Additional code cells
def clean_code(cell):
    return str(cell).replace('\\n', '\n')


def preprocess_text_sc1(document):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()
    return document


def sample_cells(cells, n):
    cells = [clean_code(cell) for cell in cells]
    if n >= len(cells):
        return [cell[:200] for cell in cells]
    else:
        results = []
        step = len(cells) / n
        idx = 0
        while int(np.round(idx)) < len(cells):
            results.append(cells[int(np.round(idx))])
            idx += step
        assert cells[0] in results
        if cells[-1] not in results:
            results[-1] = cells[-1]
        return results


def get_features(df, n_sample):
    features = dict()
    df = df.sort_values('rank').reset_index(drop=True)
    for idx, sub_df in tqdm(df.groupby('id')):
        features[idx] = dict()
        total_md = sub_df[sub_df.cell_type == 'markdown'].shape[0]
        code_sub_df = sub_df[sub_df.cell_type == 'code']
        total_code = code_sub_df.shape[0]
        # sample cells
        codes = sample_cells(code_sub_df.source.values, n_sample)
        features[idx]['total_code'] = total_code
        features[idx]['total_md'] = total_md
        features[idx]['codes'] = codes
    return features


if not os.path.exists('../input/data'):
    os.mkdir('../input/data')

n_sample = 30

df = pd.read_feather('../input/data/df.feather')

# all
df_mark = df[df['cell_type'] == 'markdown'].reset_index(drop=True)
df_mark['source'] = df_mark['source'].map(preprocess_text_sc1)
df_mark.to_feather('../input/data/df_mark.feather')

df_fts = get_features(df, n_sample)
json.dump(df_fts, open(f'../input/data/df_fts_code{n_sample}.json', 'wt'))

# train valid
# NVALID = 0.1  # size of validation set
# splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)
# train_ind, val_ind = next(splitter.split(df, groups=df['ancestor_id']))
# train_df = df.loc[train_ind].reset_index(drop=True)
# valid_df = df.loc[val_ind].reset_index(drop=True)

# train_df.to_feather('../input/data/train.feather')
# valid_df.to_feather('../input/data/valid.feather')

# # Base markdown dataframes
# train_df_mark = train_df[train_df['cell_type'] == 'markdown'].reset_index(drop=True)
# valid_df_mark = valid_df[valid_df['cell_type'] == 'markdown'].reset_index(drop=True)
# train_df_mark.to_feather('../input/data/train_mark.feather')
# valid_df_mark.to_feather('../input/data/valid_mark.feather')

# train_fts = get_features(train_df, n_sample)
# valid_fts = get_features(valid_df, n_sample)
# json.dump(train_fts, open(f'../input/data/train_fts_code{n_sample}.json', 'wt'))
# json.dump(valid_fts, open(f'../input/data/valid_fts_code{n_sample}.json', 'wt'))

# # sample
# NVALID = 0.2  # size of validation set
# splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)
# train_ind_sample, val_ind_sample = next(splitter.split(valid_df, groups=valid_df['ancestor_id']))
# train_df_sample = valid_df.loc[train_ind_sample].reset_index(drop=True)
# valid_df_sample = valid_df.loc[val_ind_sample].reset_index(drop=True)
# train_df_sample.to_feather('../input/data/train_sample.feather')
# valid_df_sample.to_feather('../input/data/valid_sample.feather')

# # Base markdown dataframes
# train_df_mark_sample = train_df_sample[train_df_sample['cell_type'] == 'markdown'].reset_index(drop=True)
# valid_df_mark_sample = valid_df_sample[valid_df_sample['cell_type'] == 'markdown'].reset_index(drop=True)
# train_df_mark_sample.to_feather('../input/data/train_mark_sample.feather')
# valid_df_mark_sample.to_feather('../input/data/valid_mark_sample.feather')

# train_fts_sample = get_features(train_df_sample, n_sample)
# valid_fts_sample = get_features(valid_df_sample, n_sample)
# json.dump(train_fts_sample, open(f'../input/data/train_fts_sample_code{n_sample}.json', 'wt'))
# json.dump(valid_fts_sample, open(f'../input/data/valid_fts_sample_code{n_sample}.json', 'wt'))
