import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import common_utils as utils
import config


def render_image(temp: np.ndarray,
                 mz: np.ndarray,
                 abd: np.ndarray,
                 step_pos: np.ndarray,
                 temp_step=6,
                 temp_query_range=10,
                 prob_smooth=6,
                 max_mz=99,
                 max_temp=999
                 ):
    min_temp = 30

    # temp_step = 6
    max_temp_id = (max_temp - min_temp) // temp_step

    # temp_query_range = 10

    res = np.zeros((max_mz + 1, max_temp_id), dtype=np.float32) - 1

    step_temp = [np.mean(v) for v in np.split(temp, step_pos)][1:-1]
    temp_bands = [[] for t in range(max_temp + temp_query_range + 1)]  # temp: list of steps
    for step, t in enumerate(step_temp):
        t = math.floor(t)
        if 0 <= t < len(temp_bands):
            temp_bands[t].append(step)

    for temp_id in range(max_temp_id):
        t = min_temp + temp_id * temp_step
        src_steps = []
        src_steps_p = []
        for band in temp_bands[max(0, t - temp_query_range):t + temp_query_range + 1]:
            for step in band:
                src_steps.append(step)
                src_steps_p.append(1.0 / (prob_smooth + abs(t - step_temp[step])))

        if not len(src_steps):
            continue

        src_steps_p = np.array(src_steps_p)
        src_step = np.random.choice(src_steps, p=src_steps_p / src_steps_p.sum())

        for i in range(step_pos[src_step], step_pos[src_step + 1]):
            res[mz[i], temp_id] = abd[i]

    return res


def check_render_image():
    sample_id = 'S0754'
    sample_id = 'S0001'

    # t = pd.read_csv(f'../data/train_features/{sample_id}.csv')
    # t = t.loc[(t['m/z'] % 1 < 0.3) | (t['m/z'] % 1 > 0.7)].reset_index(drop=True)
    # t['m/z'] = t['m/z'].round().astype(int)
    # t = t.loc[t['m/z'] < 100].reset_index(drop=True)
    # t = t.loc[t['temp'] < 1000].reset_index(drop=True)
    # step_pos = np.where(np.diff(t['m/z'].values, prepend=0) < 0)[0]
    #
    # t["abundance_minsub"] = t.groupby(["m/z"])["abundance"].transform(
    #     lambda x: (x - x.quantile(0.05))
    # )

    t = pd.read_csv(f'../data/train_features_pp/{sample_id}.csv')
    step_pos = np.where(np.diff(t['m/z'].values, prepend=0) < 0)[0]

    abd = t['abundance_sub_q20'].values
    abd = abd / abd.max()

    mz = t['m/z'].values.astype(int)

    with utils.timeit_context('render img'):
        for i in range(100):
            p = render_image(temp=t['temp'].values, mz=mz, abd=abd, step_pos=step_pos)

    p[4, :] *= 0.1
    p = p / p.max()

    p = np.clip(p, 1e-5, 1.0)
    p = np.log10(p)

    # plt.hist(p)
    # plt.figure()

    plt.imshow(p)
    plt.show()


class MarsSpectrometryDataset(Dataset):
    def __init__(self, fold, is_training,
                 dataset_type='train',
                 output_pytorch_tensor=True,
                 temp_step=6,
                 temp_query_range=10,
                 prob_smooth=6,
                 he_scale=1.0,
                 he_var=0.0,
                 min_clip=1e-5,
                 max_mz=99,
                 max_temp=999,
                 norm_to_one=True,
                 mz_var=0.0,
                 mix_aug_prob=0.0,
                 sub_bg_prob=0.0,
                 normalize_max=True,  # normalize after bg sub
                 normalize_m_separately=False,
                 log_space=True
                 ):

        self.max_temp = max_temp
        self.max_mz = max_mz
        self.output_pytorch_tensor = output_pytorch_tensor
        self.is_training = is_training
        self.fold = fold
        self.temp_step = temp_step
        self.prob_smooth = prob_smooth
        self.temp_query_range = temp_query_range
        self.he_scale = he_scale
        self.he_var = he_var
        self.min_clip = min_clip
        self.log_space = log_space
        self.norm_to_one = norm_to_one
        self.mz_var = mz_var
        self.mix_aug_prob = mix_aug_prob
        self.normalize_max = normalize_max
        self.sub_bg_prob = sub_bg_prob
        self.normalize_m_separately = normalize_m_separately

        if not is_training:
            self.mix_aug_prob = 0

        self.sample_ids = []
        self.metadata = pd.read_csv("../data/metadata.csv", index_col='sample_id')

        if dataset_type == 'train':
            folds = pd.read_csv('../data/folds_v4.csv')

            if fold == 'sam':
                if is_training:
                    folds = folds[folds.instrument_type != 'sam_testbed']
                else:
                    folds = folds[folds.instrument_type == 'sam_testbed']
            else:
                fold = int(fold)
                if is_training:
                    folds = folds[folds.fold != fold]
                else:
                    folds = folds[folds.fold == fold]

            self.sample_ids = list(folds.sample_id.values)
            self.labels = pd.concat([
                pd.read_csv('../data/train_labels.csv', index_col='sample_id'),
                pd.read_csv('../data/val_labels.csv', index_col='sample_id')
            ], axis=0)
            # self.labels = pd.read_csv('../data/train_labels.csv', index_col='sample_id')
        elif dataset_type == 'val':
            self.sample_ids = list(self.metadata[self.metadata.split == 'val'].index.values)
            self.labels = pd.read_csv('../data/val_labels.csv', index_col='sample_id')
        elif dataset_type == 'test_val':
            self.sample_ids = list(self.metadata[self.metadata.split != 'train'].index.values)
            self.labels = pd.DataFrame(index=self.sample_ids)
            for col in config.CLS_LABELS:
                self.labels[col] = 0.0

        self.samples_data = {
            sample_id: pd.read_pickle(f'../data/features_pp_v4/{sample_id}.csv.pkl')
            for sample_id in self.sample_ids
        }

        print(f'Training {is_training}, samples: {len(self.sample_ids)} ')

    def __len__(self):
        return len(self.sample_ids)

    def render_item(self, sample_id):
        t = self.samples_data[sample_id]
        if self.max_mz < 100:
            t = t[t['m/z'] <= self.max_mz].reset_index(drop=True)
        step_pos = np.where(np.diff(t['m/z'].values, prepend=0) < 0)[0]
        prob_sub = np.random.rand(4)
        prob_sub = prob_sub / prob_sub.sum()

        sub_bg = False
        if self.sub_bg_prob > 0:
            sub_bg = np.random.rand() < self.sub_bg_prob

        if sub_bg:
            abd = (
                    t['abundance_sub_bg_sub_min'].values * prob_sub[0] +
                    t['abundance_sub_bg_sub_q5'].values * prob_sub[1] +
                    t['abundance_sub_bg_sub_q10'].values * prob_sub[2] +
                    t['abundance_sub_bg_sub_q20'].values * prob_sub[3]
            )
        else:
            abd = (
                    t['abundance_sub_min'].values * prob_sub[0] +
                    t['abundance_sub_q5'].values * prob_sub[1] +
                    t['abundance_sub_q10'].values * prob_sub[2] +
                    t['abundance_sub_q20'].values * prob_sub[3]
            )

        # if self.normalize_max:
        #     abd = abd / abd.max()

        mz = t['m/z'].values.astype(int)
        p = render_image(temp=t['temp'].values, mz=mz, abd=abd, step_pos=step_pos,
                         temp_step=self.temp_step,
                         temp_query_range=self.temp_query_range,
                         prob_smooth=self.prob_smooth,
                         max_temp=self.max_temp,
                         max_mz=self.max_mz
                         )

        if self.normalize_m_separately:
            p[:, :] /= np.clip(np.max(p, axis=1, keepdims=True), 1e-5, 1e5) + 5e-4

        if self.mz_var > 0:
            p[:, :] *= 2 ** np.random.normal(loc=0, scale=self.mz_var, size=(p.shape[0], 1))

        # p = p / p.max() * (2 ** np.random.normal(0, 0.05))
        p = np.clip(p, 0, 16.0)
        return p

    def __getitem__(self, item):
        sample_id = self.sample_ids[item]
        labels = self.labels.loc[sample_id]
        label_values = [labels[k] for k in config.CLS_LABELS]
        label_values = np.array(label_values, dtype=np.float32)
        metadata = self.metadata.loc[sample_id]

        p = self.render_item(sample_id)

        sample_id2 = ''
        label_values2 = np.zeros_like(label_values)
        if self.mix_aug_prob > 0 and np.random.rand() < self.mix_aug_prob:
            if label_values.sum() < 0.5:  # for empty labels, don't mix noise with peaks
                p = p * 1e-2 * (2 ** np.random.normal(0.0, 1.0))

            sample_id2 = np.random.choice(self.sample_ids)
            labels2 = self.labels.loc[sample_id2]
            label_values2 = np.array([labels2[k] for k in config.CLS_LABELS], dtype=np.float32)
            p2 = self.render_item(sample_id2)
            if label_values2.sum() < 0.5:
                p2 = p2 * 1e-2 * (2 ** np.random.normal(0.0, 1.0))

            if np.random.rand() > 0.5:
                p = np.maximum(p.copy(), p2.copy())
            else:
                p = (p.copy() + p2.copy()) / 2
            label_values = np.maximum(label_values, label_values2)

        p[4, :] *= self.he_scale * (2 ** np.random.normal(0, self.he_var + 1e-3))
        if self.normalize_max:
            p = p / p.max()
        p = p * (2 ** np.random.normal(0, 0.05))

        # print(self.min_clip)
        # utils.print_stats('p', p)
        p = np.clip(p, self.min_clip, 1.0)

        if self.log_space:
            p = np.log10(p)
            if self.norm_to_one:
                p = 1.0 + p / abs(np.log10(self.min_clip))

        p = p.astype(np.float32)

        if self.output_pytorch_tensor:
            p = torch.from_numpy(p)
            label_values = torch.from_numpy(label_values)

        res = {
            'item': item,
            'image': p,
            'sample_id': sample_id,
            'sample_id2': sample_id2,
            'label': label_values,
            'label2': label_values2,
            'instrument_type': metadata['instrument_type']
        }

        return res


def check_dataset():
    ds = MarsSpectrometryDataset(
        fold=0, is_training=True, output_pytorch_tensor=False,
        normalize_max=True,
        normalize_m_separately=False,
        he_scale=0.1,
        mz_var=0.1,
        # dataset_type='val',
        min_clip=1e-4,
        temp_step=12,
        temp_query_range=16,
        prob_smooth=12)

    ds.sample_ids = ds.sample_ids[::-1]

    for i, sample in enumerate(ds):
        print(sample['item'], sample['sample_id'], sample['instrument_type'], sample['label'])
        utils.print_stats('image', sample['image'])

        plt.imshow(sample['image'])
        plt.xlabel('T')
        plt.ylabel('m/z')
        plt.show()


def check_aug():
    ds = MarsSpectrometryDataset(
        fold=0, is_training=True, output_pytorch_tensor=False,
        he_scale=0.1,
        he_var=0.0,
        temp_step=12,
        temp_query_range=16,
        prob_smooth=12,
        norm_to_one=False,
        min_clip=1e-4,
        mix_aug_prob=0.0,
        mz_var=1.0,
        sub_bg_prob=0.5
    )

    sample_id = 'S0491'

    for i in range(100):
        sample = ds[ds.sample_ids.index(sample_id)]
        print(sample['item'], sample['sample_id'], sample['sample_id2'], sample['label'], sample['label2'])
        utils.print_stats('image', sample['image'])

        plt.imshow(sample['image'])
        plt.show()


def check_performance():
    ds = MarsSpectrometryDataset(fold=0, is_training=True, output_pytorch_tensor=False,
                                 he_scale=0.1,
                                 he_var=0.0,
                                 temp_step=12,
                                 temp_query_range=16,
                                 prob_smooth=12,
                                 norm_to_one=False,
                                 min_clip=1e-4,
                                 mix_aug_prob=0.0,
                                 mz_var=1.0,
                                 sub_bg_prob=0.5
                                 )
    print()
    with utils.timeit_context('predict 100'):
        for i, sample in tqdm(enumerate(ds), total=len(ds)):
            pass


if __name__ == '__main__':
    # check_render_image()
    check_dataset()
    # check_aug()
    # check_performance()

