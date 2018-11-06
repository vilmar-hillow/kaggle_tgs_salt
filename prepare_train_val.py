from prepare_data import data_path
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


def get_split_old(fold):

    train_path = data_path / 'train' / 'images'

    train_file_names = np.array([x for x in train_path.glob('*')])

    splits = KFold(n_splits=4, shuffle=True, random_state=7777).split(train_file_names)

    folds = {}
    for i, indices in enumerate(splits):
        folds[i] = indices

    train, val = train_file_names[folds[fold][0]], train_file_names[folds[fold][1]]
    return train, val


def get_split(fold):
    train_path = data_path / 'train' / 'images'
    n_fold = 5
    depths = pd.read_csv(str(data_path / 'depths.csv'))
    depths.sort_values('z', inplace=True)
    depths.drop('z', axis=1, inplace=True)
    depths['fold'] = (list(range(n_fold)) * depths.shape[0])[:depths.shape[0]]

    val_df = depths[depths["fold"] == fold]
    val_ids = list(val_df["id"])
    val_ids = [train_path / (x + ".png") for x in val_ids if (train_path / (x + ".png")).is_file()]

    train_df = depths[depths["fold"] != fold]
    train_ids = list(train_df["id"])
    train_ids = [train_path / (x + ".png") for x in train_ids if (train_path / (x + ".png")).is_file()]

    return train_ids, val_ids


def get_test():
    test_path = data_path / 'test' / 'images'

    return [x for x in test_path.glob('*')]
