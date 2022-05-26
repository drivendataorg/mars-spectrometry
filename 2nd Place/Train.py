#!/usr/bin/env python
# coding: utf-8

# !ls ../data


# !cp -r ../data -d data/


# !pip install pipreqs


get_ipython().system('rm -r Train.py')


get_ipython().system("jupyter nbconvert --no-prompt --to script 'Train.ipynb' ")


get_ipython().system('pipreqs --ignore "./.ipynb_checkpoints" --force ./ ')








MODEL_PATH = 'models'
PREDS_PATH = 'preds'





import random, datetime
random.seed(datetime.datetime.now().microsecond)

RUN = [ 'lgb']


HELIUM = False #True


X_MEAN = False #random.choice([True, False])
FULL = False # random.choice([True, False])
SMOOTH_SERIES = False;# random.choice([True, False])


MODEL = ''
STRIDE1, STRIDE2 = (2, 2)

BATCH_SIZE = 12
ACT = 'LeakyReLU'
RRELU_EARLY = 0
RRELU_MID = 0
RRELU_FINAL = 0


TEMP_WIDTH = 1000
TEMP_SPLIT = 25
TEMP_OFFSET = random.randrange(80, 120)


X3_PWR = random.choice([ 0.5, 0.8, 1])

WD = random.choice([
                            1e-5,
                         1e-3, 
                        1e-4,
                   ])
MZ_CHANNELS = random.choice([ 
                             256, 
])

MZ_DROPOUT = random.choice([0.05, ])

STEM_DROPOUT = 0; RC_DROPOUT = 0; INPUT_DROPOUT = 0;

X_INPUTS = random.choice([['x2'], 
                          ['x1', 'x2'],
                          ['x1', 'x2', 'x3'],
                         ])

if random.random() < 1/3:
    STEM_DROPOUT = random.choice([
                                0.1, 
                               ])
elif random.random() < 1/2:
    RC_DROPOUT = random.choice([
                                 0.1, 
                               ])
else:
    INPUT_DROPOUT = 0.1 
    
print(TEMP_SPLIT, TEMP_OFFSET, MZ_CHANNELS, MZ_DROPOUT,
      ''.join(sorted(X_INPUTS)) if len(X_INPUTS) < 3 else '',
      INPUT_DROPOUT, STEM_DROPOUT, RC_DROPOUT,
     WD, ACT, BATCH_SIZE, MODEL, STRIDE1, STRIDE2 )


RANDOM_TRIM = random.choice([ 5, 10, 20, 50, ])
print(RANDOM_TRIM)


PREV_OFFSET = random.randrange(0, 100)








import os


import pandas as pd
import numpy as np
import scipy


import datetime, time
import pickle


from collections import defaultdict


import scipy.signal as ss


from joblib import Parallel, delayed


import boto3


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 5)


from IPython.display import display


pd.set_option('display.max_columns', 100)


plt.rcParams['figure.max_open_warning'] = 200


meta = pd.read_csv('data/metadata.csv')
# meta


submission = pd.read_csv('data/submission_format.csv')
# submission#.tail()





supp = pd.read_csv('data/supplemental_metadata.csv')
# supp.tail()


train_labels = pd.read_csv('data/train_labels.csv')
# train_labels


val_labels = pd.read_csv('data/val_labels.csv')


train_labels = pd.concat((train_labels, val_labels), ignore_index = True)





commerical = meta[meta.instrument_type.str.startswith('comm')]#.sample_id


y = train_labels.set_index('sample_id')








def runMetric(y, y_preds, verbose = False):
    l = np.array([log_loss(y.iloc[:, i], y_preds.reindex(y.index).iloc[:, i] )
                          for i in range(y.shape[1])])
    if verbose: display(l.round(3).tolist())
    return round(np.mean(l), 4)


class MultiLabelStratifiedKFold:
    def __init__(self, n_splits=5, random_state=None, full = False):
        self.n_splits = n_splits
        self.random_state = random_state
        self.full = full
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
        
    def split(self, X, y=None, groups=None, verbose = 0):
        n_cols = y.shape[1]
        if self.full:
            y_full = pd.DataFrame(0, y.index, 
                                  np.arange(0, y.shape[1] * (y.shape[1] +1  ) //2), dtype =np.float32)
            y_full.iloc[:, :y.shape[1]] = y.copy()
            ctr = y.shape[1]
            for i in range(y.shape[1]):
                for j in range(y.shape[1]):
                    if i >= j: continue;
                    y_full.iloc[:, ctr] = y.iloc[:, i] * y.iloc[:, j]; ctr +=1
            y = y_full
        freq = y.mean(axis = 0 ).values
        fold_ids = [[] for i in range(self.n_splits)]
        folds = [ np.zeros(y.shape) for i in range(self.n_splits)]
        lwt =  (len(y) / 5) ** -0.5 / ( 1 + y.sum(axis = 1).mean())
        y = y.sample(frac = 1, random_state = self.random_state).astype(np.int32)
        # print(y)
        def safeLog(x, eps = 1e-2): return np.log(np.array(x) * (1-  eps) + eps)
        
        for sample_id, row in zip(y.index, y.values):
        # for sample_id, row in y.sample(frac = 1, random_state = self.random_state).iterrows():
            for idx in range(len(folds)):
                folds[idx][len(fold_ids[idx])] = row#.values
            cts = [np.sum(f[:len(fold_ids[idx])], axis = 0) for idx, f in enumerate(folds)]
            pre_means = [cts[idx] / len(fold_ids[idx]) 
                                if len(fold_ids[idx]) > 0 else np.zeros(y.shape[1]) 
                         for idx, f in enumerate(folds)]
            means = [(cts[idx] + row ) / (len(fold_ids[idx]) +1)
                                   for idx, f in enumerate(folds)]
            pre_scores = ( ( safeLog(pre_means) - safeLog(freq) )** 2).sum(axis = 1) 
            post_scores = ( ( safeLog(means) - safeLog(freq)) ** 2).sum(axis = 1)
            # print(pre_scores)
            # print(post_scores)
            psd =  pre_scores.std() 
            delta_score = post_scores - pre_scores + ( [ psd
                                  * lwt *
                                  ( len(fold_ids[idx]) 
                                       + f[:len(fold_ids[idx]), :n_cols].sum() )
                                       for idx, f in enumerate(folds)])
            # print(delta_score)
            i = np.argmin(delta_score)# for 
            fold_ids[i].append(sample_id)
        if verbose > 0:
            display([np.sum(f, axis = 0) for f in folds])
            print([np.sum(f) for f in folds])
            print([len(f) for f in fold_ids])
        return [(list(set(y.index) - set(f)), f) for f in fold_ids]


from sklearn.metrics import log_loss








# s3 = boto3.client('s3')


# for folder in os.listdir('data'):
#     if os.path.isfile('data/' + folder) or not folder.endswith('features'): continue;
#     print(folder)
#     for file in os.listdir('data/' + folder):
#         print(file); #break;
#         s3.upload_file(os.path.join('data/', folder, file),
#                             'projects-v',  os.path.join('Mars/data/', folder, file) )
#         # break;











train_labels.iloc[:, 1:].mean().sort_values()[::-1]





# !unzip data/val_features.zip -d data/
# !unzip data/test_features.zip -d data/
# !unzip data/train_features.zip -d data/
# !unzip data/supplemental_features.zip -d data/








c = 4





samples = train_labels[(train_labels.iloc[:, c] == 1)
                       & (train_labels.iloc[:, 1:].sum(axis = 1) == 1)].sample_id.tolist()


tc = meta[ (meta.split == 'train')
           &  meta.instrument_type.str.startswith('comm')
         ].iloc[:20]


i = 19
sample = tc.iloc[i]


def loadSample(sample):
    try:
        file_path = 'data/' + sample.features_path
        s3 = boto3.client('s3')
        f = s3.get_object(Bucket = 'projects-v', Key = 'Mars/' + file_path)['Body']    
        s = pd.read_csv(f)
    except:
        s = pd.read_csv(file_path)
    return s


def cleanSample(s):
    s = s[( s['m/z'] % 1 == 0) & (s['m/z'] < 100) 
              & ( HELIUM | (s['m/z'] != 4) )]
    vc = s['m/z'].value_counts()
    valid = vc[vc.values > 0.1 * vc.quantile(0.98)]
    s = s[s['m/z'].isin(valid.index)]
    return s


def smooth(x, w = 10):
    s = np.concatenate(([x[0]] * (w//2), x, [x[-1]] * (w - w//2 - 1 )))
    d = np.hanning(w)

    y = np.convolve(d/d.sum(), s, mode='valid')
    return y

def smoothSeries(sp, return_window = False):
    w_true = int(40 * np.abs(sp.diff()).median() / np.abs(sp.diff()).mean())
    w = max(min(w_true, 50), 9)
    sp = pd.Series(smooth(sp.values, w = w), sp.index)
    if return_window: return sp, w_true
    else: return sp


def getSpectra(s, smooth = SMOOTH_SERIES):
    spectra = {}
    for m, sp in s.set_index('temp').groupby('m/z'):
        if m % 1 != 0: continue
        if m == 4 and not HELIUM: continue;
        a = sp.abundance 
        if smooth and a.sum() != 0: a = smoothSeries(a)
        b = a - np.quantile(a.values, 0.01 * np.exp(random.random() - 0.5))
        spectra[m] = b#.clip(0, None)
    smax = max([sp.max() for m, sp in spectra.items() if m != 4])
    for m, sp in spectra.items():
        spectra[m] /= smax
    return spectra


def getStats(spectra):
    total = sum([sp.sum() for m, sp in spectra.items() 
                     if m != 4 or HELIUM])##.sum()
    # print(total)
    summeans = sum([sp.mean() for m, sp in spectra.items() 
                    if m != 4 or HELIUM])

    means = {m:round(sp.mean(), 5) for m, sp in spectra.items()}
    totals = {m:round(sp.sum()/total, 4) for m, sp in spectra.items()}
    raw_peak = {m:round(sp.max(), 4) for m, sp in spectra.items()}
       
    peak = {m:round(sp.max()/summeans, 2) for m, sp in spectra.items()}
    peak_to_mean = {m: ( round(sp.max()/sp.mean(), 1) if sp.sum() > 0 else 0)
                        for m, sp in spectra.items()}
    peak_temp = {m: sp.idxmax() for m, sp in spectra.items()}
    peak_temp = {m: round(-500 if peak_temp[m] <= sp.index[0]
                         else (5000 if peak_temp[m] >= sp.index[-1] else peak_temp[m]), 1 )
                     for m, sp in spectra.items()}
    
    width_at_half = {}
    for m, sp in spectra.items():
        pk = sp[(sp > 0.5 * sp.max())]
        width_at_half[m] = pk.index.max() - pk.index.min()
        
    jitter = {m: sp.diff().std() for m, sp in spectra.items()}
    
    
    feats = {'means': means, 'totals': totals, 'raw_peak': raw_peak,
                'peak_temp': peak_temp, 'peak': peak, 'peak_to_mean': peak_to_mean,
                'width_at_half': width_at_half, 'jitter': jitter
                # 'width_at_20pct': width_at_20pct,
                # 'width_at_20pct': width_at_80pct
            }
    
    
    # for t in range(-100, 1100, 100):
    #     feats['a{}'.format(t)] = { m: sp[(sp.index >= m) & (sp.index < m + 100)].mean() 
    #                                    for m, sp in spectra.items() }

    return feats


get_ipython().run_cell_magic('time', '', 's = loadSample(sample)\ns = cleanSample(s)\nspectra = getSpectra(s)')


stats = getStats(spectra)
means, totals, raw_peak, peak_temp, peak, peak_to_mean = [stats[k] for k in 
                                             ['means', 'totals', 'raw_peak', 'peak_temp', 'peak', 'peak_to_mean']]








def interpolate(sp, start = 0, end = TEMP_WIDTH, offset = TEMP_OFFSET):
    xi = np.arange(start + offset, end + offset, 1.0)
    sp = sp - sp[(sp.index >= xi.min() ) & (sp.index < xi.max())
                 ].quantile(0.01 * np.exp(random.random() - 0.5))
    # sp = sp.clip(0, None)
    yi = scipy.interpolate.interp1d(sp.index, 
                                    sp.values,
                                  fill_value = 0.,# (sp.values[0], sp.values[-1]), 
                                    bounds_error = False)(
        xi)
    return pd.Series(yi, xi)


def getImage(spectra, deg = TEMP_SPLIT, ):#offset = TEMP_OFFSET):
    si = interpolate(list(spectra.values())[0])
    x = np.zeros((len(si)//deg, 100), dtype = np.float32)
    # x = []
    for m in range(100):
        if m not in spectra: continue;
        si = interpolate(spectra[m])
        si = si.values.reshape(-1, deg).mean(axis = 1)
        x[:, m] = si
    # x = np.stack(x, axis = 1)
    return x











trim_ranges =  [] 
trim_ranges =  [(-1000, 2000) 
               for i in range((len(meta) - len(trim_ranges)))]


random.seed(datetime.datetime.now().microsecond)
random.shuffle(trim_ranges)





def loadImage(sample, trim_range):
    s = loadSample(sample)
    s = cleanSample(s)
    s = s[(s.temp > trim_range[0]) & (s.temp < trim_range[1])]
    
    spectra = getSpectra(s)

    x = getImage(spectra)
    return x


get_ipython().run_cell_magic('time', '', 'x = loadImage(sample, trim_ranges[-4])')








get_ipython().run_cell_magic('time', '', 'x1 = x / ( 100 * x.mean() \n                                                       if X_MEAN else 1\n                                               );\nx2 = x / (x.sum(axis = 1) + 1e-12) [:, None]\nx3 = x / (x.sum(axis = 0) ** X3_PWR + 1e-12) [None, :]')


for arr in [x1, x2, x3, ]:
    print(arr.std())





plt.matshow(x1)


plt.matshow(x2); #plt.colorbar()


plt.matshow(x3)





valid = [e for e, _ in sorted(list(totals.items()), key = lambda x: -x[-1]) if _ > 0.001]
# valid





from collections import defaultdict


def getPeaks(sp, ):
    p = ss.find_peaks(sp, height = sp.mean(), distance = 8, width = 1,
                         prominence = sp.max() * 0.03
                     )
    
    if len(p[0]) == 0:
        # if sp.idxmax() == sp.index.max():
        return np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
    
    # display(p)
    peak_temps = sp.iloc[p[0]].index.values.round(1)
    peak_heights = p[1]['peak_heights'] #/ p[1]['peak_heights'].max() )
    prominences =  p[1]['prominences'] #/ p[1]['prominences'].max() ).round(2)
    pfilter = prominences > 0.2 * (prominences.sum() - prominences.max() ) 
    # print(pfilter)
    
    return peak_temps[pfilter], peak_heights[pfilter], prominences[pfilter]


# totals


def runPeaks(spectra, raw_peak, verbose = 0):
    peaks = defaultdict(dict)
    for m, t in sorted(list(raw_peak.items()), key = lambda x: -x[-1]):
        if m == 4 and not HELIUM: continue;
        if m == 1: continue;
        if t > 0.001:
            if verbose: print(m)
            pt, ph, pp = getPeaks(spectra[m])            
            for i, idx in enumerate(np.argsort(pp)[::-1][:3]):
                peaks['ppt{}'.format(i)][m] = pt[idx]
                peaks['ppp{}'.format(i)][m] = pp[idx]
                peaks['pph{}'.format(i)][m] = ph[idx]
            # for i, idx in enumerate(np.argsort(ph)[::-1][:3]):
            #     peaks['pht_{}'.format(i)][m] = pt[idx]
            #     peaks['phh_{}'.format(i)][m] = ph[idx]

            if verbose : display(p); print()
    return dict(peaks)





def getPrevalence(s, interval = 50, scaling = 'mean'):
    pdict = defaultdict(dict)
    offset = PREV_OFFSET
    for t in range(offset, 1000 + offset, interval):
        prev = s[(s.temp >= t) & (s.temp <= t + interval)].groupby(['m/z']).abundance.mean()
        prev = prev / (prev.max() if scaling == 'max' else prev.sum())
        for r in prev.iteritems():
            pdict['prevalence{}{}'.format(scaling, t )][r[0]] = r[1]
    return dict(pdict)





train_labels[(train_labels.basalt == 1)].iloc[:, 1:].sum(axis = 1).value_counts()# == 1)].sum()[1:]


train_labels[(train_labels.basalt == 1) & (train_labels.iloc[:, 1:].sum(axis = 1) == 1)].sum()[1:]








peaks = getPeaks(spectra[32])





def getBasics(sample):
    s = loadSample(sample)
    s = cleanSample(s)
    p =(s.temp.max() - s.temp.min()) / (s.time.max() - s.time.min())
    # spectra = getSpectra(s)
    random.seed(datetime.datetime.now().microsecond)
    spectra = getSpectra(s[(s.temp >= s.temp.min() + RANDOM_TRIM * random.random())
      & (s.temp < s.temp.max() - RANDOM_TRIM * random.random())])

    stats = getStats(spectra)
    feats = {}
    for arr in stats.keys():#['means', 'totals', 'raw_peak', 'peak_temp', 'peak', 'peak_to_mean']: 
        feats.update({'{}_{}'.format(arr, i): stats[arr].get(i, -500 if 'temp' in arr else 0) for i in range(100)})
    
    peaks = runPeaks(spectra, stats['raw_peak'])
    for arr in peaks.keys():
        feats.update({'{}_{}'.format(arr, i): peaks[arr].get(i, -500 if 'ppt' in arr else 0) for i in range(100)})

    prev = getPrevalence(s, 50, 'mean')
    for arr in prev.keys():
        feats.update({'{}_{}'.format(arr, i): prev[arr].get(i, 0) for i in range(100)})

    feats['pace'] = p
    feats['mean'] = s.abundance.mean()
    feats['max'] = s.abundance.max()
    feats['obs'] = s['m/z'].value_counts().median()
    return feats








get_ipython().run_cell_magic('time', '', "items = []\nwindows = []\nfor sample in tc.itertuples():\n    s = loadSample(sample)\n    s = cleanSample(s)\n    spectra = getSpectra(s[(s.temp >= s.temp.min() + RANDOM_TRIM * random.random())\n      & (s.temp < s.temp.max() - RANDOM_TRIM * random.random())])\n    # spectra = getSpectra(s)\n    stats = getStats(spectra)\n    # peaks = runPeaks(spectra, stats['raw_peak'])\n    # feats = getBasics(sample)\n    # items.append((stats, peaks, feats))\n    \n    for m, sp in spectra.items():\n        # if m!= 32: continue;\n        if m == 4 and not HELIUM: continue;\n        \n        if stats['totals'][m] > 0.001:\n            \n            # sp, w = smoothSeries(sp, return_window = True)\n            # windows.append(w)            \n            sp.plot(label = m, c = (m / 100, (m % 20)/20, (m % 3) / 3), figsize = (10, 4))\n            \n    plt.xlim(-50, 1050)\n    plt.legend()\n    plt.figure()\n    # plt.yscale('log')")








# ### Run

get_ipython().run_cell_magic('time', '', "r = Parallel(os.cpu_count())(delayed(loadImage)(sample, trim_range)\n                             for sample, trim_range in \n                              zip( meta[::30 if 'nn' not in RUN else 1].itertuples(),\n                                      trim_ranges))\n\nimages = dict(zip(meta.sample_id, r))")


get_ipython().run_cell_magic('time', '', "r = Parallel(os.cpu_count())(delayed(getBasics)(sample) \n     for sample in meta[::10 if 'lgb' not in RUN else 1].itertuples())\n\nfeats = dict(zip(meta.sample_id, r))\n\nfeat_df = pd.DataFrame.from_dict(feats, orient = 'index', dtype = np.float32)")


temp_cols = [c for c in feat_df.columns if 'temp' in c or 'ppt' in c]
non_temp_cols = [c for c in feat_df.columns if c not in temp_cols]
feat_df.loc[:, temp_cols] = feat_df[temp_cols].fillna(-500)
feat_df.loc[:, non_temp_cols] = feat_df[non_temp_cols].fillna(0)#-500)
#.astype(np.float#isnull().sum()


(feat_df > 0).sum().plot()


feat_df.head()





# ### Modeling

# %%time
folds = MultiLabelStratifiedKFold(5, 
                         random_state = datetime.datetime.now().microsecond,# full = True
                                 ).split(
                    None, y, verbose = 1)

[[len(e) for e in f] for f in folds]


y_preds = []
for train_idxs, test_idxs in folds:
    y_preds.append(
        pd.DataFrame(
            np.repeat( 
                np.expand_dims(y.loc[train_idxs].mean(axis = 0).values, axis = 0),
                    len(test_idxs), axis = 0 ),
            index = test_idxs,
            columns = y.columns) )
y_preds = pd.concat(y_preds)


runMetric(y, y_preds)








# ### Neural Network

import math
import torch
import pytorch_lightning as pl

from torch import nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from adamp import AdamP, SGDP


sam = ( meta.set_index('sample_id').instrument_type.str.startswith('sam')
          * 1) .to_dict()





class SpectrumDataset(Dataset):
    def __init__(self, images, y):
        self.x, self.y = [], []; self.sam = []
        for sample_id, row in y.iterrows():
            x = images[sample_id]
            xs = []
            if 'x1' in X_INPUTS: xs.append(x / ( 100 * x.mean() 
                                                       if X_MEAN else 1
                                               ));
            if 'x2' in X_INPUTS: xs.append(x / (x.sum(axis = 1) + 1e-12) [:, None] )
            if 'x3' in X_INPUTS: xs.append(x / (x.sum(axis = 0) ** X3_PWR + 1e-12) [None, :] )
            self.x.append(torch.Tensor(np.stack(xs)))#.dtype
            self.y.append(torch.Tensor(row))
            self.sam.append(sam[sample_id])#meta.sample

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.sam[idx]


class Conv1dSame(torch.nn.Conv1d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih = x.size()[-1]

        pad = self.calc_same_pad(i=ih, 
                                 k=self.kernel_size[0], 
                                 s=self.stride[0], 
                                 d=self.dilation[0])

        if pad > 0:#  or pad_w > 0:
            x = F.pad(
                x, [pad // 2, pad - pad // 2]
            )
        return F.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )





stem_channels = 8 #if (TEMP_SPLIT > 10 ) else 4
stride1 = STRIDE1
stride2 = STRIDE2
reduce_channels = 16
mz_channels = MZ_CHANNELS
mz_dropout = MZ_DROPOUT
split_stem = False #if (TEMP_SPLIT > 10 ) else True

batch_size = BATCH_SIZE

optimizer = SGDP
lr = 0.1 * (batch_size / 12) ** (1/2)#  * LR_MULT
wd = WD

dropout = 0.5

input_dropout = INPUT_DROPOUT
stem_dropout = STEM_DROPOUT
rc_dropout = RC_DROPOUT





class SpectrumModel(pl.LightningModule):
    def __init__(self, stem_channels = 8, reduce_channels = 16, 
                 mz_channels = 128, split_stem = False,
                input_channels = 3):
        self.preds = []
        self.epoch = -1; # self.n = 0
        self.split_stem = split_stem
        
        super().__init__()
        
        self.input_dropout = nn.Dropout(input_dropout)
        
        self.dilations = [1, 2, 5, 10, 20, 50, 100]
        if self.split_stem:
            self.stems = nn.ModuleList(
                [nn.ModuleList([Conv1dSame(1, stem_channels, kernel_size = 3, 
                                           stride = stride1, # if TEMP_SPLIT > 10 else 5 , 
                            dilation = d, bias = False)  for d in self.dilations])
                    for ich in range(input_channels)] )
        else:
            self.stems = nn.ModuleList([
                Conv1dSame(1, stem_channels, kernel_size = 3, stride = stride1, 
                                      dilation = d, bias = False) 
                        for d in self.dilations])
        
        
        self.bn1 = nn.GroupNorm(len(self.dilations), stem_channels * len(self.dilations) * input_channels)
        self.a1 = nn.RReLU(0, RRELU_EARLY) if RRELU_EARLY > 0 else getattr(nn, ACT)()# nn.LeakyReLU()
        
        self.stem_dropout = nn.Dropout(stem_dropout)
        
        self.conv2 = Conv1dSame(stem_channels * len(self.dilations) * input_channels, reduce_channels, 
                                kernel_size = 3, stride = stride2,
                                    bias = False)
        self.bn2 = nn.GroupNorm(8, reduce_channels)
        self.a2 = nn.RReLU(0, RRELU_MID) if RRELU_MID > 0 else getattr(nn, ACT)()# nn.LeakyReLU()

        self.rc_dropout = nn.Dropout(rc_dropout)
        
        self.mz_dropout = nn.Dropout(mz_dropout)
        
        self.linear_mz = nn.Linear(x1.shape[1], mz_channels, bias = False)
        self.bn3 = nn.GroupNorm(8, mz_channels)
        self.a3 = nn.RReLU(0, RRELU_FINAL) if RRELU_FINAL > 0 else getattr(nn, ACT)()# nn.LeakyReLU()
        
#         self.merge_conv = nn.Conv1d(512, 64, kernel_size = 1,
#                                  bias = False, groups = 32)
        
#         self.bn4 = nn.GroupNorm(8, 64)
#         self.a4 = nn.LeakyReLU()
        
        self.final_dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(reduce_channels * len(x1) // (stride1 * stride2)
                                              * mz_channels, y.shape[1])
        
    def forward(self, x): 
        x0 = torch.clone(x)
        b, ch, temp, mz = x.shape #  (b, 3, temp, mz) 
        
        
        if self.split_stem:
            x = torch.permute(x, (1, 0, 3, 2)) #  (3, b, mz, temp)
            x = x.reshape((ch, -1, 1, x.shape[-1])) #  (3, b * mz, 1, temp) 
            x = torch.stack( [ stem(x[ich]) for ich, stems_ich in enumerate(self.stems)
                                    for stem in stems_ich])
            #  (ch * num_dilations, b * mz, stem_channels, temp // 2)
            x = x.reshape((-1, len(self.dilations), 
                               b, mz,
                                   x.shape[-2],  x.shape[-1]))
            # (ch, num_dilations,  b, mz, stem_channels, temp//2)
            x = torch.permute(x, (2, 3, 1, 0, 4, 5))
            # (b, mz, num_dilations, ch, stem_channels, temp // 2)
            # print(x.shape)# torch.Size([32, 100, 5, 3, 4, 50])
            x = x.reshape(b * mz,
                              x.shape[2] *  x.shape[3] * x.shape[4],
                                  x.shape[5] )
            #  (b * mz,  num_dilations * ch * stem_channels,  temp// 2)
            # print(x.shape) # torch.Size([3200, 60, 50])

        else:
            # Conv1d along multiple strides 
            # x[:, random.randrange(0, 3), :, :] = 0
            x = torch.permute(x, (0, 1, 3, 2)) #  (b, 3, mz, temp)
            x = x.reshape((-1, x.shape[-1])).unsqueeze(1) #  (b * 3 * mz, 1, temp) 

            x = torch.permute(x, (1, 0, 2))
            x = self.input_dropout(x)
            x = torch.permute(x, (1, 0, 2)) #  (b * 3 * mz, 1, temp) again 
            
            # print(x.shape)  # torch.Size([9600, 1, 100])
            x = torch.stack( [stem(x) for stem in self.stems] )
            #  (num_dilations, b * 3 * mz, 4ch, temp // 2)
            # print(x.shape) # torch.Size([5, 9600, 4, 50])
            x = x.reshape((x.shape[0], 
                               b, ch, mz,
                                   x.shape[-2],  x.shape[-1]))
            # (num_dilations,  b, 3, mz, 4ch, temp//2)
            # print(x.shape) # torch.Size([5, 32, 3, 100, 4, 50])
            x = torch.permute(x, (1, 3, 0, 2, 4, 5))
            # (b, mz, num_dilations, 3, stem_channels, temp // 2)
            # print(x.shape)# torch.Size([32, 100, 5, 3, 4, 50])
            x = x.reshape(b * mz,
                              x.shape[2] * x.shape[3] * x.shape[4],
                                  x.shape[5] )
            #  (b * mz,  num_dilations * 3 * stem_channels,  temp// 2)
            # print(x.shape) # torch.Size([3200, 60, 50])
            
        # print(x.shape) # torch.Size([3200, 60, 50])
        #  (b * mz,  num_dilations * 3 * stem_channels,  temp// 2)    
        x = self.a1(self.bn1(x))                                
        x = self.stem_dropout(x)
                                
        # Conv1d again, and reduce channels 
        x = self.conv2(x)
        # (b * mz, 16ch, temp // 4 )
        # print(x.shape)   # torch.Size([3200, 16, 25])
        _, tch, tr = x.shape#[1]
        x = self.a2(self.bn2(x))
        x = self.rc_dropout(x)
                
        x = x.reshape((b, mz, x.shape[1], x.shape[2]))
        x = torch.permute(x, (0, 2, 3, 1))
        # (b, 16ch, temp // 4, mz)
        # x = torch.cat((x, 
        #                torch.mean(x, 2, keepdim = True),
        #                torch.max(x, 2, keepdim = True).values,
        #                torch.max(x, 2, keepdim = True).indices ),
        #                  dim = 2)
        
        x = x.reshape((-1, mz))
        
        x = self.mz_dropout(x)
        x = self.linear_mz(x)
        # (b * 16 t-ch * temp//4, 32 mz-ch)
        # print(x.shape) # torch.Size([12800, 32])
        x = self.a3(self.bn3(x))
        
        x = x.reshape((b, tch, -1, x.shape[-1]))
        # (b, 16 tch, temp//4 + 3, 32 mzch)
        # x = torch.cat((x, 
        #                torch.mean(x, 2, keepdim = True),
        #                torch.max(x, 2, keepdim = True).values ),
        #                  dim = 2)
        # (b, 16 tch, temp//4 + 5, 32 mzch)
        # print(x.shape) # torch.Size([32, 16, 30, 32])
        b, tch, tf, mzch = x.shape
        
        x = x.reshape((b, - 1)) # (b, mzch * tch * tf)
        
        
        x = self.final_dropout(x)
        x = self.final_linear(x)
        
        return x
    
        

    def training_step(self, batch, batch_idx):
        x, y, s = batch
        yp = self.forward(x)
        wt = 1 - s
        wt = wt / torch.sum(wt) * len(wt)
        loss = nn.BCEWithLogitsLoss(
            weight = wt.unsqueeze(1)
                                   )(yp, y)

        
        self.log("train_loss", loss)
        return loss
    
    def on_validation_epoch_start(self):
        self.epoch_preds = []
        
    def validation_step(self, batch, batch_idx):
        x, y, s = batch
        yp = self.forward(x)
        self.epoch_preds.append((y, yp, s))
        
    def on_validation_epoch_end(self):
        y = torch.cat([e[0] for e in self.epoch_preds], axis = 0)
        yp = torch.cat([e[1] for e in self.epoch_preds], axis = 0)
        s = torch.cat([e[2] for e in self.epoch_preds], axis = 0)
        
        self.epoch += 1
        if len(self.preds) > 0:
            # self.n += 1
            a = 1 / ( self.epoch - 5)
            self.preds = self.preds * ( 1 - a ) + yp * a
            yp = self.preds
            
        
        elif len(y) == len(test_dataset) and self.epoch > 5:
            self.preds = yp; #self.n = 1;
            
        scored = torch.isnan(y).sum(axis = 1) == 0
        wt = 1 + 8 * s[scored] # 9x weight
        wt = wt / torch.sum(wt) * len(wt)
        loss = nn.BCEWithLogitsLoss(
            weight = wt.unsqueeze(1)
        )(yp[scored], y[scored] )
        self.log("val_loss", loss)
        print(loss)
        
        
        if save_preds and self.epoch in [10, 20, 30, 50, 70, 100]:
            pred_df = pd.DataFrame(self.preds, inference_idxs, 
                                       train_labels.columns[-10:])
            model_name = 'nn_base8' + ('h' if HELIUM else '')
            pickle.dump(pred_df, 
                open('{}/{}{}{}b{}{}{}_temp{}{}_o{}{}{}{}_mz{}{}{}{}_folds{}_rs{}_fold{}_epoch{}'.format(
                   PREDS_PATH,
                    model_name, '',#  'trt' if TRIM_RANGES else '',
                    '-wd{}'.format(wd) if wd != 1e-2 else '',
                        BATCH_SIZE,
                            ''.join(sorted(X_INPUTS)) if len(X_INPUTS) < 3 else '',
                            'x3p{}'.format(X3_PWR) if X3_PWR != 1 else '',
                        TEMP_SPLIT, 's{}{}'.format(stride1, stride2) if stride1 * stride2 != 4 else '',
                            (TEMP_OFFSET // 50) if TEMP_WIDTH == 900 else (TEMP_OFFSET//40),
                            'id{}'.format(INPUT_DROPOUT) if INPUT_DROPOUT > 0 else '',
                            'sd{}'.format(STEM_DROPOUT) if STEM_DROPOUT > 0 else '',
                            'rd{}'.format(RC_DROPOUT) if RC_DROPOUT > 0 else '',
                            MZ_CHANNELS,
                                ('-d{}'.format(MZ_DROPOUT) if MZ_DROPOUT > 0 else ''),
                                ACT if ACT != 'LeakyReLU' else '',
                                '-{},{},{}'.format(RRELU_EARLY, RRELU_MID, RRELU_FINAL)
                                        if RRELU_EARLY + RRELU_MID + RRELU_FINAL > 0 else '',
                             n_folds, #'full' if STRAT_FULL else 'basic', 
                                folds_rs, fold_idx, self.epoch), 'wb') )    
        
        
        return loss

    def configure_optimizers(self):
        return optimizer(self.parameters(), 
                                    lr=lr, weight_decay = wd, )# momentum = 0.75)





save_preds = True


sam_ids = set(meta[meta.instrument_type.str.startswith('sam')].sample_id)





get_ipython().run_cell_magic('time', '', "all_preds = []\nwhile 'nn' in RUN:\n    n_folds = 5\n    folds_rs = datetime.datetime.now().microsecond\n    folds = MultiLabelStratifiedKFold(n_folds, \n                         random_state = folds_rs,\n                                  full = FULL# random.choice([True, False])\n                                 ).split(\n                    None, y, verbose = 1)\n\n\n    for fold_idx, (train_idxs, test_idxs) in enumerate(folds):\n        train_idxs = list(set(train_idxs) - set(sam_ids));         \n        train_dataset = SpectrumDataset(images, y.loc[train_idxs])\n        \n        \n        inference_idxs = list( set(meta.sample_id) - set(train_idxs) ) \n        \n        test_dataset = SpectrumDataset(images, y.reindex(inference_idxs))\n        train_loader = DataLoader(train_dataset, batch_size=batch_size,\n                                      shuffle=True, drop_last = True, \n                                          num_workers = os.cpu_count())\n        test_loader = DataLoader(test_dataset, batch_size=128, \n                                     shuffle=False,\n                                 num_workers = os.cpu_count())\n\n        model = eval('SpectrumModel' + MODEL)(stem_channels, reduce_channels, \n                                              mz_channels, split_stem, \n                                             input_channels = train_dataset[0][0].shape[0])        \n        trainer = pl.Trainer(gpus=0, max_epochs = 101 if TEMP_SPLIT > 10 else 51,#   if mz_dropout == 0 else 51, \n                             enable_progress_bar = False,\n                        # log_every_n_steps = len(train_dataset) // batch_size,\n                                enable_checkpointing=False, logger = False)                        \n        trainer.fit(model, train_loader, test_loader, )\n        all_preds.append(model.preds)\n        # break;\n    # break;")











# ### LGB Model

if 'lgb' in RUN: assert set.issubset(set(y.index), set(feat_df.index))





import lightgbm as lgb
import datetime


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, KFold


from sklearn.linear_model import LogisticRegression


from sklearn.preprocessing import StandardScaler





def noise(x):    
    x = x.copy()
    rng =  np.random.default_rng(seed = datetime.datetime.now().microsecond)
    
    temp_cols = [c for c in x.columns if ('temp' in c or 'ppt' in c) and 'cross' not in c]
    non_temp_cols = [c for c in x.columns if c not in temp_cols]# 'temp' in c or 'ppt' in c]

    
    x.loc[:, temp_cols] =  x[temp_cols] + np.where((x[temp_cols] > -500) 
                                                       & (x[temp_cols] < 5000),
                                rng.standard_normal(x[temp_cols].shape, np.float32) 
                                                   * 5 * random.random()
                                                   , 0)
    
    

    return x


get_ipython().run_cell_magic('time', '', 'noise(feat_df)')





key = '''
oxychlorine: 31, 32
oxalate: 28, 29, 44
iron_oxide: 20, 30
chloride: 36
carbonate: 43, 44, 42
basalt: 18, 30
sulfide: 64, 48
sulfate: 64, 48'''

key_elements = {}
for row in key.split('\n'):
    if ':' not in row: continue;
    key_elements[row.split(':')[0]] = [int(i) for i in row.split(':')[-1].split(', ')]
assert set(key_elements.keys()).issubset(y.columns)

display(key_elements)





def getElementPairs(target, x_train, y_train, relative = False, verbose = False):    
    ranked_elements = (
       x_train.loc[ y_train[y_train[target] == 1].index, 
                    [c for c in x_train.columns if c.startswith('means_')]
               ].median(axis = 0)
      
      / ( x_train.loc[ y_train[ (y_train[target] == 0) #& (y_train.sum(axis =1) > 0)
                            ].index, 
                    [c for c in x_train.columns if c.startswith('means_')]
               ].median(axis = 0)#.sort_values()[::-1]#[:10].index
            if relative else 1)
     ).sort_values()[::-1][1:]#[:10].index
    # print(ranked_elements)

    top_elements = [int(f.split('_')[-1]) for f in ranked_elements[:10].index]
    if verbose: print('Top Elements for {}: '.format(target), top_elements)
    
    added_elements = key_elements.get(target, [])
    top_elements = added_elements + [e for e in top_elements if e not in added_elements] 
    if verbose: print('All Elements for {}: '.format(target), top_elements)
        

    element_pairs = []
    for eidx1, e1 in enumerate(top_elements):
        for eidx2, e2 in enumerate(top_elements):
            if eidx2 <= eidx1 or eidx1 + eidx2 > 10:
                continue;
            element_pairs.append((e1, e2))

    return element_pairs





def addElementPairFeatures(x, element_pairs):
    new_cols = {}
    for e1, e2 in element_pairs:
        for idx1 in range(2):
            for idx2 in range(2):
                new_cols['ppt{}{}_cross_{}_{}'.format(idx1, idx2, e1, e2)] =                     np.where(  ( x['ppt{}_{}'.format(idx1, e1)] > -500 ) &
                               ( x['ppt{}_{}'.format(idx2, e2)] > -500 ),
                        x['ppt{}_{}'.format(idx1, e1)] - x['ppt{}_{}'.format(idx2, e2)], np.nan)

                new_cols['ppp{}{}_cross_{}_{}'.format(idx1, idx2, e1, e2)] =                     np.where(  ( x['ppp{}_{}'.format(idx1, e1)] > 0 ) &
                               ( x['ppp{}_{}'.format(idx2, e2)] > 0 ),
                        x['ppp{}_{}'.format(idx1, e1)] / x['ppp{}_{}'.format(idx2, e2)], np.nan)

                new_cols['pph{}{}_cross_{}_{}'.format(idx1, idx2, e1, e2)] =                     np.where(  ( x['pph{}_{}'.format(idx1, e1)] > 0 ) &
                               ( x['pph{}_{}'.format(idx2, e2)] > 0 ),
                        x['pph{}_{}'.format(idx1, e1)] / x['pph{}_{}'.format(idx2, e2)], np.nan)

        new_cols['means_cross_{}_{}'.format(e1, e2)] =             np.where(  ( x['means_{}'.format( e1)] > 0 ) &
                       ( x['means_{}'.format( e2)] > 0 ),
                x['means_{}'.format(e1)] / x['means_{}'.format(e2)], np.nan)

        new_cols['peak_to_mean_cross_{}_{}'.format(e1, e2)] =             np.where(  ( x['peak_to_mean_{}'.format( e1)] > 0 ) &
                       ( x['peak_to_mean_{}'.format( e2)] > 0 ),
                x['peak_to_mean_{}'.format(e1)] / x['peak_to_mean_{}'.format(e2)], np.nan)
        
    extra_cols_df = pd.DataFrame(new_cols, index = x.index)

    x = pd.concat((x, extra_cols_df), axis = 1)
    return x








n_bags = 1


[[c for c in feat_df.columns if c.startswith(target + '_')]
            for target in y.columns]


[len([c for c in feat_df.columns if c.startswith(target + '_')]) == 6
            for target in y.columns]


y.columns








save_preds = True


targets = y.columns
targets


NO_SAM = False


sam_ids = set(meta[meta.instrument_type.str.startswith('sam')].sample_id)


lgb_params = {
    'n_estimators': [100, 150, 200, ],
    'learning_rate': [0.03, 0.05, 0.07, ],
    'num_leaves': [3, 5, 7, 10, 14, 20,  ],
    'min_child_weight': [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3],
    'min_child_samples': [5, 10, 20, 30, 50, 70 , ],
    'reg_alpha': [ 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, ],
    'reg_lambda': [ 1e-4, 1e-3, 1e-2, 0.1, 1, 10,],
    'subsample': [ 0.8, 0.9, 1.0],
    'subsample_freq': [1],
    'colsample_bytree': [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'colsample_bynode': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
}


get_ipython().run_cell_magic('time', '', "while 'lgb' in RUN:\n    n_folds = 5\n    folds_rs = datetime.datetime.now().microsecond\n    folds = MultiLabelStratifiedKFold(n_folds, \n                         random_state = folds_rs,\n                                 ).split(\n                    None, y, verbose = 1)\n\n\n    for fold_idx, (train_idxs, test_idxs) in enumerate(folds):\n        if NO_SAM: \n            train_idxs = list(set(train_idxs) - set(sam_ids)); \n            print('removing SAM from training')\n            \n        start = time.time()\n        inference_idxs = list( set(meta.sample_id) - set(train_idxs) ) \n        \n        rdfs = []; \n        models = {c: [] for c in targets}\n        y_preds = {c: [] for c in targets}\n        epairs = {c: [] for c in targets}\n\n\n        for tidx, target in enumerate(targets):\n            x_train, y_train = feat_df.loc[train_idxs], y.loc[train_idxs]\n            x_test, y_test = feat_df.reindex(inference_idxs), y.reindex(inference_idxs)\n            # print(x_train.shape)\n            element_pairs = getElementPairs(target, x_train, y_train) \n            x_train, x_test = [addElementPairFeatures( arr, element_pairs) \n                           for arr in [x_train, x_test]]\n            # print(x_train.shape)\n            for bag_idx in range(n_bags):\n                random.seed(datetime.datetime.now().microsecond)\n                model = RandomizedSearchCV(lgb.LGBMClassifier(seed = datetime.datetime.now().microsecond,\n                                                                 ), lgb_params,\n                                           cv = StratifiedKFold(n_splits = 4, shuffle = True,#n_repeats = 1,\n                                                    random_state = datetime.datetime.now().microsecond),\n                                           n_iter = 4, n_jobs = -1,\n                                           scoring = 'neg_log_loss',\n                                           random_state = datetime.datetime.now().microsecond)\n                xt = noise(x_train)#.sample(frac = 0.9)\n                # if n_bags > 1:\n                xt.loc[:, [c for c in feat_df.columns if any(c.endswith('_{}'.format(i)) \n                                for i in random.sample(np.arange(0, 100).tolist(), k = 5))] ] = np.nan\n                xt.loc[:, random.sample(list(xt.columns), k = len(xt.columns) // 10)] = np.nan\n                xt.loc[:, [c for c in feat_df.columns if c.startswith(target + '_')]] = np.nan\n                model.fit(xt, y_train[target])#.reindex(xt.index))\n                clf = model.best_estimator_\n                models[target].append(clf)\n                epairs[target].append(element_pairs)\n\n                rdf = pd.DataFrame(model.cv_results_).sort_values('rank_test_score').drop(columns = 'params')\n                rdfs.append(rdf)\n\n                y_pred = pd.Series(clf.predict_proba(x_test)[:, 1], y_test.index)\n                y_preds[target].append(y_pred)\n\n        # collect \n        y_preds = {k: pd.concat(yp) for k, yp in y_preds.items()}\n        all_y_preds = pd.DataFrame(y_preds)\n        print(runMetric(y[targets].reindex(set(all_y_preds.index) & set(y.index)), all_y_preds))\n        \n        \n        \n        if save_preds:\n            os.makedirs(PREDS_PATH, exist_ok = True)\n            pickle.dump(all_y_preds, \n                 open('{}/{}_rtrim{}_folds{}_rs{}_fold{}'.format(\n                     PREDS_PATH,\n                     'lgb_base8' + ('nosam' if NO_SAM else ''), \n                         RANDOM_TRIM,\n                         n_folds, folds_rs, fold_idx), 'wb'))\n        print('{:.1f}s elapsed'.format(time.time() - start))\n        # break;\n    # break;")


y_pred_lgb = pd.DataFrame(y_preds)


runMetric(y, y_pred_lgb) 











# ### Stacker

files = os.listdir(PREDS_PATH)
files[::2000]





def loadFile(file):
    return pickle.load(open(PREDS_PATH + '/' + file, 'rb'))


def loadFiles(files):
    preds = []
    for file in files:
        preds.append(loadFile(file))
    preds = pd.concat(preds)
    # if 'lgb' in files[1]: preds = getLogit(preds)
    preds = preds.groupby(preds.index).mean().astype(np.float32)
    return preds





def groupFiles(files):
    groups = defaultdict(list)
    files = [((
        
        (f.split('x3p')[0] + '_temp' + f.split('_temp')[-1])
            if 'x2' in f else ''.join(f.split('x3'))  
              
              
             ).split('_rs')[0] 
            + ( ( '_epoch' + f.split('_epoch')[-1]) if 'epoch' in f else '')
                  , f) for f in files
                        
            ]
    for file in files:
        groups[file[0]].append(file[1]);
    return groups


get_ipython().run_cell_magic('time', '', "lgb_files = [f for f in files if 'lgb' in f and 'stack' not in f]\nlgb_groups = groupFiles(lgb_files)\nlgb_preds = Parallel(os.cpu_count())(delayed(loadFiles)(files)\n                        for files in lgb_groups.values())")


sorted([(k, len(v)) for k, v in lgb_groups.items()],
           key = lambda x: -x[-1])


lgb_preds = dict(zip(lgb_groups.keys(), lgb_preds))


def logit(y): return 1 / (1 + np.exp(-y))
def getLogit(y): return -np.log( 1 / y - 1)

lgb_preds = {k: getLogit(v) for k, v in lgb_preds.items()}


len(lgb_preds)


plt.matshow(list(lgb_preds.values())[0].corr())
plt.colorbar()





nn_files = [f for f in files if 'nn' in f ]
nn_groups = groupFiles(nn_files)
len(nn_groups)


nn_groups = {k.replace('-0,0,0', ''): v 
                 for k, v in nn_groups.items() if len(v) >= 4 * 5}
len(nn_groups)


nn_groups = dict(sorted([(k, v) for k, v in nn_groups.items()],
           key = lambda x: -len(x[-1])))


sorted([(k, len(v)) for k, v in nn_groups.items()],
           key = lambda x: -x[-1])





get_ipython().run_cell_magic('time', '', 'nn_preds = Parallel(os.cpu_count() * 2)(delayed(loadFiles)(files[:250])\n                        for files in nn_groups.values())')


nn_preds = dict(zip(nn_groups.keys(), nn_preds))


len(nn_preds)





all_preds = {}
all_preds.update(lgb_preds)
all_preds.update(nn_preds)





all_preds = {}
all_preds.update({k: v 
                  for k, v in lgb_preds.items()
                  if 'stack' not in k
                 })
all_preds.update({k.replace('-0,0,0', ''): v 
                  for k, v in nn_preds.items()                   
                })


n_rows =  np.median([len(v) for v in all_preds.values()])
all_preds = {k: v for k, v in all_preds.items()
                 if len(v) == n_rows}





len(all_preds)





import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)


pd.Series(
    dict(sorted([(k, runMetric(y, logit(preds)) )
         for k, preds in all_preds.items() ], key = lambda x: x[0]))
).plot(kind = 'barh', figsize = (10, 20)); plt.xlim(0.11, 0.16);


# sorted(
#     [(runMetric(y, logit(preds)), k) for k, preds in all_preds.items()],
#     key = lambda x: x[-1])


sorted(
    [(runMetric(y, logit(preds)), k) for k, preds in all_preds.items()],
    key = lambda x: x[0]
)








all_pred_dfs = {}
for target in y.columns:
    all_pred_dfs[target] = pd.DataFrame({k: all_preds[k][target] 
                                         for k in sorted(all_preds.keys())})


plt.matshow(all_pred_dfs['basalt'].corr()); plt.colorbar()





lgbs = pd.concat(lgb_preds.values())
lgb_blend = lgbs.groupby(lgbs.index).mean()


# runMetric(y, logit(lgb_blend))


blend_df = pd.concat(all_preds.values())
blend_df = blend_df.groupby(blend_df.index).mean()


runMetric(y, logit(blend_df * 0.8))





plt.matshow(blend_df.corr())
plt.colorbar()








def score(wts, x, y, reg = 1, l1_ratio = 0):
    # wsum = wts.sum()
    wts = (wts / max(wts.sum() ** 0.5, 1.0) )#.astype(np.float32)
    blend = ( x * wts[None, :]).sum(axis = 1)#.astype(np.float32)
    # print(wts)
    return ( 
        log_loss(y, logit(blend))
            + reg *( (wts ** 2).sum() + l1_ratio * np.abs(wts).sum()) )


def optimize(x, y, reg = 1, l1_ratio = 0, tol = 1e-4 ):
    wts = scipy.optimize.minimize(
    score, np.ones(x.shape[1]) / x.shape[1],#len(x.columns), 
        tol = tol,
    args=(x, y, reg, l1_ratio), 
    bounds=[(0, 1) for i in range(x.shape[1])],#len(x.columns))],
    ).x
    return wts / max(wts.sum() ** 0.5, 1.0)


import sklearn


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold


class CLR(sklearn.base.BaseEstimator):
    def __init__(self, reg = 1.0, l1_ratio = 0, tol = 1e-4):
        self.reg = reg
        self.l1_ratio = l1_ratio
        self.classes_ = np.array((0, 1))
        self.tol = tol
    
    def fit(self, X, y):
        wts = optimize(X.values, y.values, 
                           self.reg, self.l1_ratio, self.tol)
        self.wts = wts #/ max(wts.sum(), 1)# * 0.9
        # print(self.wts.sum())
        
    def predict(self, X):
        return logit((X * self.wts).sum(axis = 1)).values

    def predict_proba(self, X):
        preds = self.predict(X)
        return np.stack(((1-preds), preds)).T
    
    # def clone(self):
    #     return self.copy()
    
    # def get_params(self):





lr_params = {'reg': [  3e-5, 1e-4, 3e-4, 0.001, 0.003, 
                         0.1, 0.3, 1, 3],
             # 'intercept_scaling':[ 0.1, 1, 10],
                'l1_ratio': [0.0, 0.01, 0.03, 0.1, 0.3, 0.5,   ]
            }


get_ipython().run_cell_magic('time', '', "rdfs = []# {target: [] for target in y.columns}\ny_preds = {target: [] for target in y.columns}\nmodels = {target: [] for target in y.columns}\nall_wts = {target: [] for target in y.columns}\nfor target, pred_df in all_pred_dfs.items():\n    print(target); \n    # pred_df = logit(pred_df)# + np.random.normal(0, 0.1, size = pred_df.shape)#.iloc[:, :: 3]\n    # pred_df = logit(pred_df)\n    for bag_idx in range(6):\n        n_folds = 5 # random.randrange(4, 6)\n        folds_rs = datetime.datetime.now().microsecond\n        folds = MultiLabelStratifiedKFold(n_folds, \n                             random_state = folds_rs,\n                        full = random.choice([ True, False])\n                                     ).split(\n                        None, \n            y, #y.reindex(list(set(meta[meta.instrument_type == 'commercial'].sample_id) & set(y.index)))\n            verbose = 0)\n\n\n        for fold_idx, (train_idxs, test_idxs) in enumerate(folds):\n            start = time.time()\n            inference_idxs = list( set(meta.sample_id) - set(train_idxs) ) \n\n            random.seed(datetime.datetime.now().microsecond)\n                \n            model = RandomizedSearchCV(\n                CLR(), lr_params,\n                    cv = StratifiedKFold(\n                              n_splits = random.randrange(4, 6),\n                                # n_repeats = random.choice([1, 2]),\n                                shuffle = True,\n                                random_state = datetime.datetime.now().microsecond\n                        ),\n                                    n_iter = random.randrange(3, 5),#choice([3, 4]),\n                                    n_jobs = -1,\n                            scoring = 'neg_log_loss',\n                            random_state = datetime.datetime.now().microsecond)\n            pdf = pred_df.drop(columns = random.sample(list(pred_df.columns),\n                            k = int( (0.4 + 0.2 * random.random())\n                                        * len(pred_df.columns)) ) )\n            early_cols =  [c for c in pdf.columns if 'epoch10' in c\n                                          or 'epoch20' in c or 'epoch30' in c]\n            pdf = pdf.drop(columns = random.sample(early_cols,\n                               k = int(0.7 * len(early_cols))))\n            \n            \n            model.fit(pdf.loc[train_idxs], \n                      y.loc[train_idxs, target])\n            clf = model.best_estimator_\n            # break;\n            if bag_idx == 0: print(clf)\n            models[target].append(clf)\n            all_wts[target].append(pd.Series(clf.wts, pdf.columns))\n            # epairs[target].append(element_pairs)\n\n            rdf = pd.DataFrame(model.cv_results_\n                        ).sort_values('rank_test_score').drop(\n                                    columns = 'params')\n            rdfs.append(rdf)\n\n            y_pred = pd.Series(clf.predict(\n                        pdf.loc[inference_idxs]),#[:, 1], \n                                   inference_idxs)\n            y_pred\n            y_preds[target].append(y_pred)\n        # break;\n    # break;\n        # display(rdf)\n    print()")


y_preds = {k: pd.concat(yp) for k, yp in y_preds.items()}
y_preds = {k: yp.groupby(yp.index).mean() for k, yp in y_preds.items()}


runMetric(y, pd.DataFrame(y_preds) )




# 0.1406 -- run3 
# 0.1372 -- epochs + pure offset - 20 point gain 
# 0.1356 -- xsam, o2, and 512/768/etc.  - run4 
# 0.1339 -- drop-mz, batch4, lighter wd
# 0.1310 -- wd1e-4 
# 0.1295 -- floating weight sum [0.1333 0.85-scaled avg !!!]
# 0.1292 -- sans stride and models b/c (0.1293 without mz512 -- etc.)
# 0.1287 -- d0.05 and wd1e-3:   -- run5x (0.1248 !!! 4 targets changed (!))
# 0.1251 -- various forms of dropout, etc. [channels dropout mostly]
#                [0.1262 without id/sd/rd] [0.1270 without xdrop/xd]
#                [0.1258 without single-ch] #
# 0.1223 -- !!! with more epochs for single/dual-ch and wd1e-4;#  (!!!)
# 0.1207 -- !!! sqrt scaling for x3 !!!   (0.1181 )
# reorder of convs/stride1, batch, lr, target re-wt, smooth, etc.: UNCh# 0.1036 -- !!
# 0.1008 -- !!!
# 0.1004 
# 0.1001



model_weights = {}
for k, v in all_wts.items():
    df = pd.concat(v)
    df = df.groupby(df.index).mean().reindex(pred_df.columns)
    model_weights[k] = df / df.sum()
model_weights =  pd.DataFrame(model_weights, index = pred_df.columns)


[c.startswith('ox') or c.endswith('silicate') for c in y.columns]


plt.matshow(model_weights.T)
plt.colorbar()








model_weights.mean(axis = 1).sort_index().round(3).sort_values()[::-1][:20]





combo_weights = model_weights.groupby(
    model_weights.index.to_series().apply(
        lambda x: '_'.join(x.split('_')[:3])).values).sum()
combo_weights.mean(axis = 1).sort_index().round(3).plot(kind = 'barh',                                                                                              
                                                    figsize = (10, 8));


combo_weights = model_weights.groupby(
    model_weights.index.to_series().apply(
        lambda x: '_'.join(x.split('_')[:4])).values).sum()
combo_weights.mean(axis = 1).sort_index().round(3).plot(kind = 'barh',                                                                                              
                                                    figsize = (10, 12));


combo_weights = model_weights.groupby(
    model_weights.index.to_series().apply(
        lambda x: '_'.join(x.split('_')[:5])).values).sum()
combo_weights.mean(axis = 1).sort_index().round(3).plot(kind = 'barh',                                                                                              
                                                    figsize = (10, 15));


combo_weights = model_weights#.groupby(
    # model_weights.index.to_series().apply(
    #     lambda x: '_'.join(x.split('_')[:6])).values
# ).sum()
combo_weights.mean(axis = 1).sort_index().round(3).plot(kind = 'barh',                                                                                              
                                                    figsize = (10, 30));





# # SAM samples (!!!)
# yy = y.loc[list(set(meta[meta.instrument_type.str.startswith('sam')].sample_id)
#                & set(y.index))]
# round(-(yy * np.log(pd.DataFrame(y_preds).reindex(yy.index))
#     + (1 - yy ) * np.log(1 - pd.DataFrame(y_preds).reindex(yy.index))).mean().mean(), 4)


lgbs = pd.concat([v for k, v in lgb_preds.items() if 'nosam' in k])
lgb_blend = lgbs.groupby(lgbs.index).mean()


blend_df = pd.concat([v for k, v in all_preds.items() 
                      if 'epoch100' not in k and 'epoch70' not in k
                      and 'x1' not in k and 'x2' not in k
                      and 'lgb' not in k
                         ])
blend_df = blend_df.groupby(blend_df.index).mean()


# SAM samples (!!!)
sam_total = 0.95
nn_wt = 0.25
lgb_wt = sam_total - nn_wt; 
yy = y.loc[list(set(meta[meta.instrument_type.str.startswith('sam')].sample_id)
               & set(y.index))]
round(-(yy * np.log(pd.DataFrame(logit(lgb_blend * lgb_wt + blend_df * nn_wt)).reindex(yy.index))
    + (1 - yy ) * np.log(1 - pd.DataFrame(logit(lgb_blend * lgb_wt + blend_df * nn_wt)).reindex(yy.index))).mean().mean(), 4)


# SAM samples (!!!)
sam_total = 0.95
nn_wt = 0.25
lgb_wt = sam_total - nn_wt; 
yy = y.loc[list(set(meta[meta.instrument_type.str.startswith('sam')].sample_id)
               & set(y.index))]
round(-(yy * np.log(pd.DataFrame(logit(lgb_blend * lgb_wt + blend_df * nn_wt)).reindex(yy.index))
    + (1 - yy ) * np.log(1 - pd.DataFrame(logit(lgb_blend * lgb_wt + blend_df * nn_wt)).reindex(yy.index))).mean().mean(), 4)


# 0.1407 (0.1325 before, possibly better WITHOUT epochs for sam);
# 0.1471 with staggger --- progressively worse with more complex methods
# back to 0.1436 with run4; 0.14x with run5
# 0.1330 - 80% lgb blend, 20% nn blend
# 0.1280 - sum to 0.85, etc.
# 0.1272 - not including x1/x2 models








# ### Compile

y_pred_df = pd.DataFrame(y_preds)
y_pred_df


sam_pred_df = logit(lgb_blend * lgb_wt + blend_df * nn_wt)
sam_pred_df


# sam_pred_df.reindex(sam_ids).plot(kind = 'hist', bins= 250);








sam_pred_df.plot(kind = 'hist', bins = 250);





y_pred_df.mean().plot()
y_pred_df.reindex(sam_ids).mean().plot()
sam_pred_df.reindex(sam_ids).mean().plot()
sam_pred_df.mean().plot()





y_pred_df.std().plot()
y_pred_df.reindex(sam_ids).std().plot()
sam_pred_df.reindex(sam_ids).std().plot()
sam_pred_df.std().plot()








assert all(y_pred_df.index == sam_pred_df.index)
assert all(y_pred_df.columns == sam_pred_df.columns)


# replace_df = pd.concat(( y_pred_df
                        
#                         .reindex(set(meta.sample_id) - sam_ids),
#            ( (y_pred_df * 2+ sam_pred_df )/3)
#                         .reindex(sam_ids)),)


# replace_df = pd.concat(( y_pred_df
                        
#                         .reindex(set(meta.sample_id) - sam_ids),
#            ( logit(getLogit( y_pred_df) + 2))
#                         .reindex(sam_ids)),)


y_pred_df = subs[-4]


replace_df = pd.concat(( y_pred_df
                        
                        .reindex(set(meta.sample_id) - sam_ids),
           logit( ( getLogit( (y_pred_df + sam_pred_df)/2)
                   + getLogit(y_pred_df.mean()).values[None, :]
                    ) / 2
                  )
                        .reindex(sam_ids)),)





y_pred_df.mean().plot()
y_pred_df.reindex(sam_ids).mean().plot()
replace_df.reindex(sam_ids).mean().plot()
sam_pred_df.reindex(sam_ids).mean().plot()





y_pred_sub = replace_df.loc[meta[meta.split != 'train'].sample_id]
y_pred_sub.index.name = 'sample_id'

sample_sub =pd.read_csv('data/submission_format.csv')


assert all(y_pred_sub.reset_index().columns == sample_sub.columns)
assert set(y_pred_sub.reset_index().index) == set(sample_sub.index)





y_pred_sub.index





val_labels = pd.read_csv('data/val_labels.csv', index_col = 0)


assert all(val_labels.columns == y_pred_sub.columns)


assert set(val_labels.index).issubset(y_pred_sub.index)


runMetric(val_labels, y_pred_sub.reindex(val_labels.index)[val_labels.columns])


runMetric(val_labels, y_pred_df.reindex(val_labels.index)[val_labels.columns])


y_pred_sub.to_csv('sub8e.csv', float_format = '%.5f')


y_pred_sub.to_csv('sub8e.csv', float_format = '%.5f')








y_pred_sub.reindex(val_labels.index)


val_labels.index





[f for f in  sorted(os.listdir()) if f.startswith('sub')]


subs = [pd.read_csv(f, index_col = 0)
            for f in sorted(os.listdir()) if f.startswith('sub')]


for col in y.columns:
    plt.matshow(pd.concat([s[col] for s in subs], axis = 1).corr())
    plt.title(col); plt.colorbar()





# y_pred_sub = pd.concat(subs).groupby('sample_id').mean()


for target, tmodels in models.items():
    for model in random.choices(tmodels, k = 3):
        lgb.plot_importance(model, importance_type = 'gain', title = target, max_num_features = 40,
                               figsize = (10, 7));







