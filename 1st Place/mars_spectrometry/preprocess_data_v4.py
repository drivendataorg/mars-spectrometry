import multiprocessing

import pandas as pd
import numpy as np
import os
import config
import pybaselines
from sklearn.model_selection import StratifiedKFold


def fill_t_bins(t, a, step=4):
    t = np.floor(t / step).astype(int)
    t_min = 30
    t_max = 1000

    t_idx = list(range(t_min // step, t_max // step))
    t_bins = np.array([(i + 0.5) * step for i in t_idx])

    bins = []
    for i in t_idx:
        values = a[t == i]
        if len(values):
            bins.append(np.mean(values))
        else:
            bins.append(np.nan)

    # bins = [np.mean(a[t == i]) for i in t_idx]

    skip_from = 0
    while skip_from < len(bins) and np.isnan(bins[skip_from]):
        skip_from += 1

    skip_to = len(bins)
    while skip_to > skip_from and np.isnan(bins[skip_to - 1]):
        skip_to -= 1

    t_bins = t_bins[skip_from:skip_to]
    bins = bins[skip_from:skip_to]
    bins = np.array(bins)

    if np.isnan(bins).sum() > 0:
        bins = pd.Series(bins).interpolate().values

    return t_bins, bins


def preprocess_features(src_fn, dst_fn, is_commercial):
    # print(src_fn)
    sample_df = pd.read_csv(src_fn)
    sample_df = sample_df.loc[sample_df['temp'] < 1000].reset_index(drop=True)
    sample_df = sample_df.loc[sample_df['temp'] > 28].reset_index(drop=True)
    sample_df = sample_df.loc[(sample_df['m/z'] % 1 < 0.3) | (sample_df['m/z'] % 1 > 0.7)].reset_index(drop=True)
    sample_df['m/z'] = sample_df['m/z'].round().astype(int)
    sample_df = sample_df.loc[sample_df['m/z'] < 100].reset_index(drop=True)

    max_abd = np.nanmax(sample_df["abundance"])
    res = []

    for m in sorted(sample_df["m/z"].unique()):
        t = sample_df[sample_df["m/z"] == m]["temp"].values
        a = sample_df[sample_df["m/z"] == m]["abundance"].values / max_abd

        t_bins, a_bins = fill_t_bins(t, a)
        if len(t_bins) < 4:
            print(f'Skip m {m} a {len(a)} t_bins {len(t_bins)} {src_fn} is commercial: {is_commercial}')
            continue

        nan_values = ~np.isfinite(a_bins)
        if nan_values.sum() > 0:
            print(f'{src_fn} nan: {nan_values.sum()}')
            t_bins = t_bins[~nan_values]
            a_bins = a_bins[~nan_values]

            if len(a_bins) < 4:
                print(f'No non null values: count {len(a_bins)} m {m} {src_fn}  is commercial: {is_commercial}')
                continue

        bkg_2 = pybaselines.whittaker.aspls(a_bins, lam=2e6)[0]
        if (~np.isfinite(bkg_2)).sum() > 0:
            print(f'Invalid bg: m {m} fn {src_fn} is commercial: {is_commercial}')
            bkg_2 = pybaselines.polynomial.modpoly(a_bins, t_bins, poly_order=1)[0]

        if (~np.isfinite(bkg_2)).sum() > 0:
            print(f'Still invalid bg: m {m} fn {src_fn} is commercial: {is_commercial}')
            bkg_2 = np.zeros_like(bkg_2) + np.nanmin(a_bins)

        if is_commercial:
            bkg_cur = np.interp(t, t_bins, bkg_2)
            t_cur = t
            a_cur = a
        else:
            bkg_cur = bkg_2.copy()
            t_cur = t_bins.copy()
            a_cur = a_bins.copy()

        a_no_bg = a_cur - bkg_cur
        a_no_bg_sub_min = a_no_bg - np.quantile(a_no_bg, 0.001)
        a_no_bg_sub_q5 = a_no_bg - np.quantile(a_no_bg, 0.05)
        a_no_bg_sub_q10 = a_no_bg - np.quantile(a_no_bg, 0.10)
        a_no_bg_sub_q20 = a_no_bg - np.quantile(a_no_bg, 0.20)

        a_sub_min = a_cur - np.quantile(a_cur, 0.001)
        a_sub_q5 = a_cur - np.quantile(a_cur, 0.05)
        a_sub_q10 = a_cur - np.quantile(a_cur, 0.10)
        a_sub_q20 = a_cur - np.quantile(a_cur, 0.20)

        for i, ti in enumerate(t_cur):
            res.append({
                'temp': ti,
                'm/z': m,
                'abundance': a_cur[i],

                'abundance_sub_min': a_sub_min[i],
                'abundance_sub_q5': a_sub_q5[i],
                'abundance_sub_q10': a_sub_q10[i],
                'abundance_sub_q20': a_sub_q20[i],

                'abundance_sub_bg': a_no_bg[i],

                'abundance_sub_bg_sub_min': a_no_bg_sub_min[i],
                'abundance_sub_bg_sub_q5': a_no_bg_sub_q5[i],
                'abundance_sub_bg_sub_q10': a_no_bg_sub_q10[i],
                'abundance_sub_bg_sub_q20': a_no_bg_sub_q20[i]
            })
            # res.append([ti, m, a_cur[i], a_no_bg[i], a_sub_min[i], a_sub_q5[i], a_sub_q10[i], a_sub_q20[i]])

    res_df = pd.DataFrame(
        res
    )

    res_df['m/z'] = res_df['m/z'].round().astype(int)

    res_df = res_df.sort_values(['temp', 'm/z'], axis=0)

    res_df.to_csv(dst_fn, index=False)
    res_df.to_pickle(dst_fn+'.pkl')


def preprocess_all_features():
    metadata = pd.read_csv("../data/metadata.csv")

    requests = []
    dst_dir = f'../data/features_pp_v4'
    os.makedirs(dst_dir, exist_ok=True)

    pool = multiprocessing.Pool(16)

    for _, row in metadata.iterrows():
        src_dir = f'../data/{row["split"]}_features'
        sample_id = row["sample_id"]
        is_commercial = row["instrument_type"] == "commercial"
        requests.append([f'{src_dir}/{sample_id}.csv', f'{dst_dir}/{sample_id}.csv', is_commercial])
        # preprocess_features(f'{src_dir}/{sample_id}.csv', f'{dst_dir}/{sample_id}.csv', is_commercial)

    pool.starmap(preprocess_features, requests)


def split_to_folds():
    RANDOM_SEED = 42
    skf = StratifiedKFold(n_splits=config.NB_FOLDS, random_state=RANDOM_SEED, shuffle=True)
    metadata = pd.read_csv("../data/metadata.csv")

    metadata = metadata[metadata.split != 'test']
    metadata['fold'] = -1
    for fold, (train_index, test_index) in enumerate(skf.split(metadata.sample_id, metadata.instrument_type)):
        print(fold, test_index)
        metadata.loc[test_index, 'fold'] = fold

    print(metadata['fold'].value_counts())
    print(metadata[metadata.instrument_type == 'sam_testbed']['fold'].value_counts())
    metadata[['sample_id', 'split', 'fold', 'instrument_type']].to_csv('../data/folds_v4.csv', index=False)


if __name__ == '__main__':
    split_to_folds()
    preprocess_all_features()

