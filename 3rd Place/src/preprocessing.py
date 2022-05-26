import numpy as np
import pandas as pd
import os
import gc
from omegaconf import OmegaConf
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import signal
from typing import Union
import warnings
warnings.simplefilter('ignore')

import multiprocessing as mp
n_cpus = mp.cpu_count()

Config = OmegaConf.load('../configs/data.yaml')

def read_csv(path):
    tmp_df= pd.read_csv(path)
    sample_id = path.split('/')[-1][:-4]
    tmp_df['sample_id'] = sample_id
    return tmp_df

def drop_frac_and_He(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop fractional m/z values, m/z values greater than 99, and the carrier gas (m/z = 4)
    
    :param df: pd.DataFrame
    :return: A dataframe with the m/z values rounded to the nearest integer, and the m/z values less
    than 100.
    """


    # drop fractional m/z values
    df = df[df["m/z"].transform(round) == df["m/z"]]
    
    # assert df["m/z"].apply(float.is_integer).all(), "not all m/z are integers"

    # # drop m/z values greater than 99
    df = df[df["m/z"] < 100]

    # drop carrier gas
    df = df[df["m/z"] != 4]

    return df


def preprocess_sdf(sdf: pd.DataFrame) -> pd.DataFrame:
    """
    It drops the fractional and He m/z values, drops noise, normalizes the abundance, and creates some
    additional features
    
    :param sdf: The dataframe containing the data
    :return: A dataframe containing the spectra with additional columns for feature engineering
    """
    
    ###### Dropping Fractional and He m/z values ######
    sdf = drop_frac_and_He(sdf)
    
    ###### Dropping noise ######
    sdf['min_abundance'] = sdf.groupby('sample_id')['abundance'].transform('min')
    
    #### Original sequence number #####
    sdf['seq'] = sdf.groupby('sample_id').cumcount()+1
    
    #### Sorting by abundance for feature engineering
    sdf = sdf.sort_values(by=['sample_id', 'abundance'], ascending=False).reset_index(drop=True)

    #### Sequence ordered by abundance #####
    sdf['seq_ord'] = sdf.groupby('sample_id').cumcount()+1

    
    sdf['abundance_orig'] = sdf['abundance']

    #### Normalizing abundance #####
    sdf['max_abundance'] = sdf.groupby('sample_id')['abundance'].transform('max')
    sdf['min_abundance'] = sdf.groupby('sample_id')['abundance'].transform('min')
    sdf['abundance'] = (sdf['abundance']-sdf['min_abundance'])/(sdf['max_abundance']-sdf['min_abundance'])
    sdf.drop(columns=['max_abundance', 'min_abundance'], inplace=True)

    sdf['seq_norm'] = sdf['seq']/sdf.groupby('sample_id')['seq'].transform('max')
    sdf['temp_norm'] = sdf['temp']/sdf.groupby('sample_id')['temp'].transform('max')
    sdf['temp_bin'] = sdf.groupby('sample_id')['temp'].apply(lambda x: pd.cut(x, 5, labels=False)).astype(np.int8)
    sdf['max_abundance_time'] = sdf['sample_id'].map(sdf.groupby('sample_id')['time'].nth(0))
    sdf['max_abundance_temp'] = sdf.groupby('sample_id')['temp'].transform('first')
    
    float32_cols = ['abundance', 'abundance_orig', 'seq_norm', 'temp', 'max_abundance_temp', 'time']
    sdf[float32_cols] = sdf[float32_cols].astype(np.float32)
    sdf['m/z'] = sdf['m/z'].astype(np.int32)

    sdf['seq_ord_norm'] = sdf['seq_ord']/sdf.groupby('sample_id')['sample_id'].transform('size')
    sdf['abundance_bin'] = sdf['abundance']//0.125

    return sdf

def get_sdf(metadata, data_dir=None):
    if data_dir is None:
        data_dir = Config.DATA_DIR

    metadata['features_path'] = metadata['features_path'].apply(lambda x: os.path.join(data_dir, x))
    res = Parallel(n_jobs=n_cpus)(delayed(read_csv)(path) for path in tqdm(metadata['features_path']))
    sdf = pd.concat(res).reset_index(drop=True)

    del res
    _ = gc.collect()

    sdf = preprocess_sdf(sdf)

    return sdf
 

def feature_engineering(df: pd.DataFrame, sdf: pd.DataFrame) -> pd.DataFrame:
    """
    To add useful features to the dataframe.
    All the feature engineering is done per sample_id, to avoid leakage and avoid using distribution of other samples incorporated into our features

    :param df: the dataframe containing the sample_id and target_label
    :param sdf: the dataframe containing the spectra for all the sample ids in df
    :return: A dataframe containing the spectra with additional columns for feature engineering
    """

    target_mapper = {
        'basalt': 0,
        'carbonate': 1,
        'chloride': 2,
        'iron_oxide': 3,
        'oxalate': 4,
        'oxychlorine': 5,
        'phyllosilicate': 6,
        'silicate': 7,
        'sulfate': 8,
        'sulfide': 9
    }

    df['target_label'] = df['target_label'].map(target_mapper)
    df['sample_len'] = df['sample_id'].map(sdf.groupby('sample_id').size())

    periods = [0.995, 0.98, 0.95, 0.9, 0.7]

    ############ START ############
    ########## The idea is to resample the abundance and temp for each m/z value to have a fixed length sequence after removing noise  ##########
    ########## A value of 25 was chosen after many trials to have a good representation of the spectra ##########

    sdft = sdf.sort_values(by='time').reset_index(drop=True)

    tmp = sdft.groupby(['sample_id', 'm/z'])['abundance'].apply(lambda x: signal.resample(x, 25))
    res = pd.DataFrame(dict(zip(tmp.index, tmp.values))).T
    res.index.names = ['sample_id', 'm/z']
    res = res.unstack('m/z').fillna(0)
    res.columns = [f'm/z_{c1}_abun_{c2}' for c1, c2 in res.columns]


    tmp2 = sdf.groupby(['sample_id', 'm/z'])['abundance'].apply(lambda x: signal.resample(x, 25))
    res2 = pd.DataFrame(dict(zip(tmp2.index, tmp2.values))).T
    res2.index.names = ['sample_id', 'm/z']
    res2 = res2.unstack('m/z').fillna(0)
    res2.columns = [f'm/z_{c1}_abun_{c2}' for c1, c2 in res2.columns]

    tmp3 = sdf.groupby(['sample_id', 'm/z'])['abundance_orig'].apply(lambda x: signal.resample(x, 5))
    res3 = pd.DataFrame(dict(zip(tmp3.index, tmp3.values))).T
    res3.index.names = ['sample_id', 'm/z']
    res3 = res3.unstack('m/z').fillna(0)
    res3.columns = [f'm/z_{c1}_abun_{c2}' for c1, c2 in res3.columns]

    tmp4 = sdft.groupby(['sample_id', 'm/z'])['temp'].apply(lambda x: signal.resample(x, 25))
    res4 = pd.DataFrame(dict(zip(tmp4.index, tmp4.values))).T
    res4.index.names = ['sample_id', 'm/z']
    res4 = res4.unstack('m/z').fillna(0)
    res4.columns = [f'm/z_{c1}_temp_{c2}' for c1, c2 in res4.columns]

    df = pd.merge(df, res, on='sample_id', how='left')
    df = pd.merge(df, res2, on='sample_id', how='left')
    df = pd.merge(df, res3, on='sample_id', how='left')
    df = pd.merge(df, res4, on='sample_id', how='left')
    
    ############ END ############
    
    
    ############ START ############
    ######## The idea is to capture the behavior of the sample in terms of its top x% of rows sorted by abundance per m/z ########
    ######## Also ratio between periods helps us with more important features ########
    
    periods = [0.001, 0.004, 0.01, 0.05, 0.1, 0.2]

    for en, i in tqdm(enumerate(periods), total=len(periods)):
        
             
        sdfn = sdf[sdf['seq_ord_norm'] <= i][['sample_id', 'm/z', 'temp', 'abundance', 'temp_bin']].reset_index(drop=True)
        _ = gc.collect()

        tmp = sdfn.groupby(['sample_id', 'temp_bin'])['m/z'].value_counts(normalize=True).unstack(['m/z', 'temp_bin']).fillna(0)
        tmp.columns = [f'top_{i}_sample_id_pct_count_m/z_temp_bin_' + '_'.join([str(e) for e in c]) for c in tmp.columns]
        df = pd.merge(df, tmp, on=['sample_id'], how='left')
        
        
        tmp = sdfn.groupby(['sample_id', 'm/z', 'temp_bin'])['abundance'].mean().unstack(['m/z', 'temp_bin']).fillna(0)
        tmp.columns = [f'sample_id_m/z_top_{i}_temp_abundance_mean' + '_'.join([str(e) for e in c]) for c in tmp.columns]
        df = pd.merge(df, tmp, on=['sample_id'], how='left')
        
        tmp = sdfn.groupby(['sample_id', 'm/z', 'temp_bin'])['abundance'].std().unstack(['m/z', 'temp_bin']).fillna(0)
        tmp.columns = [f'sample_id_m/z_top_{i}_temp_abundance_std' + '_'.join([str(e) for e in c]) for c in tmp.columns]
        df = pd.merge(df, tmp, on=['sample_id'], how='left')
    
        
        tmp = sdfn.groupby('sample_id')['m/z'].value_counts(normalize=True).unstack('m/z').fillna(0)
        
        tmp_orig_cols = [f'sample_id_top_abundances_pct_counts_m/z_{c}' for c in tmp.columns]
        tmp.columns = [f'sample_id_top_{i}_abundances_pct_counts_m/z_{c}' for c in tmp.columns]
        
        tmp_cols = tmp.columns
        tmp[f'std_all_m/z_pct_counts_top_{i}_abundances'] = tmp[tmp_cols].std(axis=1)
        tmp[f'mean_all_m/z_pct_counts_top_{i}_abundances'] = tmp[tmp_cols].mean(axis=1)
        
        tmp_orig_cols += [f'std_all_m/z_pct_counts', f'mean_all_m/z_pct_counts']
        
        
        df = pd.merge(df, tmp, on='sample_id', how='left')
    
        if en > 0:
            tmp.columns = tmp_orig_cols
            tmp_prev.columns = tmp_prev_orig_cols
            
            common_cols = np.intersect1d(tmp.columns, tmp_prev.columns)
            tmp_ratio = tmp[common_cols]/tmp_prev[common_cols]
            tmp_ratio.columns = [f'{c}_ratio_top_{i}_to_prev' for c in tmp_ratio.columns]
            df = pd.merge(df, tmp_ratio, on='sample_id', how='left')

        tmp_prev = tmp
        tmp_prev_orig_cols = tmp_orig_cols
        
        for col in ['temp', 'abundance']:
        
            tmp = sdfn.groupby(['sample_id', 'm/z'])[col].mean().unstack('m/z').fillna(0)

            tmp_orig_cols = [f'sample_id_top_abundances_{col}_mean_m/z_{c}' for c in tmp.columns]
            tmp.columns = [f'sample_id_top_{i}_abundances_{col}_mean_m/z_{c}' for c in tmp.columns]
            

            df = pd.merge(df, tmp, on='sample_id', how='left')

            if en > 0:
                tmp.columns = tmp_orig_cols
                tmp_prev.columns = tmp_prev_orig_cols

                common_cols = np.intersect1d(tmp.columns, tmp_prev.columns)
                tmp_ratio = tmp[common_cols]/tmp_prev[common_cols]
                tmp_ratio.columns = [f'{c}_ratio_top_{i}_to_prev' for c in tmp_ratio.columns]
                df = pd.merge(df, tmp_ratio, on='sample_id', how='left')

            tmp_prev = tmp
            tmp_prev_orig_cols = tmp_orig_cols
            
            
        for col in ['temp', 'abundance']:
        
            tmp = sdfn.groupby(['sample_id', 'm/z'])[col].std().unstack('m/z').fillna(0)

            tmp_orig_cols = [f'sample_id_top_abundances_{col}_std_m/z_{c}' for c in tmp.columns]
            tmp.columns = [f'sample_id_top_{i}_abundances_{col}_std_m/z_{c}' for c in tmp.columns]

            df = pd.merge(df, tmp, on='sample_id', how='left')

            if en > 0:
                tmp.columns = tmp_orig_cols
                tmp_prev.columns = tmp_prev_orig_cols

                common_cols = np.intersect1d(tmp.columns, tmp_prev.columns)
                tmp_ratio = tmp[common_cols]/tmp_prev[common_cols]
                tmp_ratio.columns = [f'{c}_ratio_top_{i}_to_prev' for c in tmp_ratio.columns]
                df = pd.merge(df, tmp_ratio, on='sample_id', how='left')

            tmp_prev = tmp
            tmp_prev_orig_cols = tmp_orig_cols
            
            
        for col in ['temp', 'abundance']:
        
            tmp_max = sdfn.groupby(['sample_id', 'm/z'])[col].max().unstack('m/z').fillna(0)
            tmp_min = sdfn.groupby(['sample_id', 'm/z'])[col].min().unstack('m/z').fillna(0)
            tmp = tmp_max-tmp_min
            
            tmp_orig_cols = [f'sample_id_top_abundances_{col}_range_m/z_{c}' for c in tmp.columns]
            tmp.columns = [f'sample_id_top_{i}_abundances_{col}_range_m/z_{c}' for c in tmp.columns]


            df = pd.merge(df, tmp, on='sample_id', how='left')

            if en > 0:
                tmp.columns = tmp_orig_cols
                tmp_prev.columns = tmp_prev_orig_cols

                common_cols = np.intersect1d(tmp.columns, tmp_prev.columns)
                tmp_ratio = tmp[common_cols]/tmp_prev[common_cols]
                tmp_ratio.columns = [f'{c}_ratio_top_{i}_to_prev' for c in tmp_ratio.columns]
                df = pd.merge(df, tmp_ratio, on='sample_id', how='left')

            tmp_prev = tmp
            tmp_prev_orig_cols = tmp_orig_cols
            
        for c in ['temp']:

            df[f'min_{c}_abundance_top_{i}'] = df['sample_id'].map(sdfn.groupby('sample_id')[c].min())
            df[f'max_{c}_abundance_top_{i}'] = df['sample_id'].map(sdfn.groupby('sample_id')[c].max())
            df[f'range_{c}_abundance_top_{i}'] = df[f'max_{c}_abundance_top_{i}'] - df[f'min_{c}_abundance_top_{i}']
            df[f'std_{c}_abundance_top_{i}'] = df['sample_id'].map(sdfn.groupby('sample_id')[c].std())
            
            if c == 'abundance':
                del df[f'max_{c}_abundance_top_{i}'], df[f'min_{c}_abundance_top_{i}'], df[f'range_{c}_abundance_top_{i}']
            
    ####### Sample-level features #######
    ###### Here again features are taken for top n rows sorted by abundance ######
            
    for i in tqdm([1, 10, 100, 1000, 10000]): 
        sdf_top_n = sdf[sdf['seq_ord'] <= i][['sample_id', 'temp', 'm/z', 'abundance']].reset_index(drop=True)
        for c in ['temp', 'm/z','abundance']:
            if i == 1 and c == 'abundance':
                continue

            df[f'min_{c}_abundance_top_{i}'] = df['sample_id'].map(sdf_top_n.groupby('sample_id')[c].min())
            if i == 1:
                continue
                
            df[f'max_{c}_abundance_top_{i}'] = df['sample_id'].map(sdf_top_n.groupby('sample_id')[c].max())
            df[f'range_{c}_abundance_top_{i}'] = df[f'max_{c}_abundance_top_{i}'] - df[f'min_{c}_abundance_top_{i}']
            if i == 2:
                continue
            df[f'std_{c}_abundance_top_{i}'] = df['sample_id'].map(sdf_top_n.groupby('sample_id')[c].std())

            if c == 'abundance':
                del df[f'max_{c}_abundance_top_{i}']
                
    ####### Here we try to capture the temperature at which a certain category of abundance occurs per m/z, and the distribution of the temperature #######
                                
    tmp = sdf.groupby(['sample_id', 'm/z', 'abundance_bin'])['temp'].std().unstack(['m/z', 'abundance_bin']).fillna(0)
    tmp.columns = [f'sample_id_temp_std_m/z_abundance_bin_' + '_'.join([str(e) for e in c]) for c in tmp.columns]
    df = pd.merge(df, tmp, on=['sample_id'], how='left')
    
    tmp = sdf.groupby(['sample_id', 'm/z', 'abundance_bin'])['temp'].mean().unstack(['m/z', 'abundance_bin']).fillna(-99999)
    tmp.columns = [f'sample_id_temp_mean_m/z_abundance_bin_' + '_'.join([str(e) for e in c]) for c in tmp.columns]
    df = pd.merge(df, tmp, on=['sample_id'], how='left')

    tmp_min = sdf.groupby(['sample_id', 'm/z', 'abundance_bin'])['temp'].min().unstack(['m/z', 'abundance_bin']).fillna(-99999)
    tmp_max = sdf.groupby(['sample_id', 'm/z', 'abundance_bin'])['temp'].max().unstack(['m/z', 'abundance_bin']).fillna(-99999)
    tmp_range = tmp_max-tmp_min
    
    tmp_min.columns = [f'sample_id_temp_minn_m/z_abundance_bin_' + '_'.join([str(e) for e in c]) for c in tmp.columns]
    tmp_max.columns = [f'sample_id_temp_max_m/z_abundance_bin_' + '_'.join([str(e) for e in c]) for c in tmp.columns]
    tmp_range.columns = [f'sample_id_temp_range_m/z_abundance_bin_' + '_'.join([str(e) for e in c]) for c in tmp.columns]
    
    df = pd.merge(df, tmp_min, on=['sample_id'], how='left')
    df = pd.merge(df, tmp_max, on=['sample_id'], how='left')
    df = pd.merge(df, tmp_range, on=['sample_id'], how='left')
    
    ###### Finally we take a normalized m/z count for each temp bin ######
    tmp = sdf.groupby(['sample_id', 'temp_bin'])['m/z'].value_counts(normalize=True).unstack(['m/z', 'temp_bin']).fillna(0)
    tmp.columns = [f'sample_id_pct_count_m/z_temp_bin_' + '_'.join([str(e) for e in c]) for c in tmp.columns]
    df = pd.merge(df, tmp, on=['sample_id'], how='left')

    return df


def create_training_and_testing_data(metadata:pd.DataFrame, labels=Union[None, pd.DataFrame]) -> pd.DataFrame:
    """
    It takes in the metadata and labels dataframes, merges them, and then converts the problem into a
    binary classification problem.
    
    :param metadata: This is the metadata dataframe that we read from the csv file.
    :param labels: The labels for the training data
    :return: A dataframe with the features and the target for each sample and each label
    """

    ##### Reading sample data
    print("Reading sample data")
    sdf = get_sdf(metadata)

    id_vars = ['sample_id', 'split', 'instrument_type', 'features_path', 'features_md5_hash']
    value_vars = ['basalt', 'carbonate', 'chloride', 'iron_oxide', 'oxalate', 'oxychlorine', 'phyllosilicate', 'silicate', 'sulfate', 'sulfide']

    #### Training Data ####
    if labels is not None:
        df = pd.merge(metadata, labels, on='sample_id', how='left')

    #### Test Data ####
    else:
        df = metadata.copy()
        for c in value_vars:
            df[c] = 0

    ##### Converting it into a binary classification problem #####
    df = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name='target_label', value_name='target')

    print("Starting Feature Engineering")
    df = feature_engineering(df, sdf)

    del sdf
    _ = gc.collect()

    return df

    
