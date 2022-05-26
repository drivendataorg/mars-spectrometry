#!/usr/bin/env python
# coding: utf-8

MODEL_PATH = 'models'
PREDS_PATH = 'preds'





import random, datetime
random.seed(datetime.datetime.now().microsecond)


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


RANDOM_TRIM = random.choice([ 5, 10, 20, 50, ])
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


# %%time
s = loadSample(sample)
s = cleanSample(s)
spectra = getSpectra(s)


stats = getStats(spectra)
means, totals, raw_peak, peak_temp, peak, peak_to_mean = [stats[k] for k in 
                                             ['means', 'totals', 'raw_peak', 'peak_temp', 'peak', 'peak_to_mean']]








def interpolate(sp, start = 0,):
    end = TEMP_WIDTH
    offset = TEMP_OFFSET
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








class SpectrumModel(pl.LightningModule):
    def __init__(self, stem_channels = 8, reduce_channels = 16, 
                 mz_channels = 128, split_stem = False,
                input_channels = 3, model_name = ''):
        self.preds = []
        self.epoch = -1; # self.n = 0
        self.split_stem = split_stem
        
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
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
        self.log("val_loss", loss, prog_bar = True)
        # print(loss)
        
        
        if save_preds and self.epoch in [10, 20, 30, 50, 70, 100]:
            pred_df = pd.DataFrame(self.preds, inference_idxs, 
                                       train_labels.columns[-10:])
            pickle.dump(pred_df, 
                open('{}/{}_epoch{}'.format(
                    PREDS_PATH, self.model_name, self.epoch), 'wb') )    
        
        
        return loss

    def configure_optimizers(self):
        return optimizer(self.parameters(), 
                                    lr=lr, weight_decay = wd, )# momentum = 0.75)





save_preds = True


sam_ids = set(meta[meta.instrument_type.str.startswith('sam')].sample_id)





model_files = [f for f in os.listdir('models') 
               if 'nn_base8a' in f and 'epoch=059' in f#'rs2' in f
]
# random.shuffle(model_files)
# model_files = model_files[:500]


len(model_files)





model_files = sorted(model_files, key = lambda x: x.split('_rs')[-1].split('_'))


weights = pd.read_csv('model_weights.csv', index_col = 0)


select_models = defaultdict(list)
for k in model_files:
    match = [w for w in weights.index if k.startswith(w + '_')]
    assert len(match) <= 1
    if len(match) == 1:
        if len(select_models[match[0]]) < 50:
            select_models[match[0]].append(k)
        else:
            pass;
    else:
        print(k)


def flatten(l):
    return [e for s in l for e in s]


model_files = flatten(list(select_models.values()))
len(model_files)





# sorted(list(nn_models))[:5]


# sorted([m.split('_rs')[0] for m in model_files if 'p0' in m])[::5]





stem_channels = 8 #if (TEMP_SPLIT > 10 ) else 4
stride1 = STRIDE1
stride2 = STRIDE2
reduce_channels = 16
mz_channels = 256
mz_dropout = 0.05
split_stem = False #if (TEMP_SPLIT > 10 ) else True

batch_size = BATCH_SIZE

optimizer = SGDP
lr = 0.1 * (batch_size / 12) ** (1/2)#  * LR_MULT
# wd = WD

dropout = 0.5

#     input_dropout = INPUT_DROPOUT
#     stem_dropout = STEM_DROPOUT
#     rc_dropout = RC_DROPOUT








def getOffset(file):
    return int(file.split('temp25_o')[-1].split('-')[0])


model_files = sorted(model_files, key = lambda x: getOffset(x))


print(len(model_files))





# %%time
print(' --- Running NN Models --- ')

model_preds = {}
PRIOR_OFFSET = -1
start = time.time()
for model_idx, model_file in enumerate(model_files):
    TEMP_OFFSET = getOffset(model_file)
    if TEMP_OFFSET != PRIOR_OFFSET:
        if model_idx > len(model_files) * 0.05:
            print('{:.0f}s of ~{:.0f}s elapsed'.format(time.time() - start,
                                                     (time.time() - start)/model_idx * len(model_files)
                                                   ))
            
        print('\nRunning models with offset of {}'.format(TEMP_OFFSET))
        inference_idxs = meta[meta.split != 'train'].sample_id.tolist()
        r = Parallel(os.cpu_count())(delayed(loadImage)(sample, trim_range)
                                     for sample, trim_range in 
                                      zip( meta[meta.split != 'train'].itertuples(),
                                              trim_ranges))
        x1 = r[0]
        images = dict(zip(meta[meta.split != 'train'].sample_id, r))
        PRIOR_OFFSET = TEMP_OFFSET

    # print(time.time() - start)
    X3_PWR = ( float('0.' + model_file.split('p0.')[-1].split('_')[0]) if 'p0.' in model_file else 1)
    X_INPUTS = ['x2'] if 'b12x2_' in model_file else (
            ['x1', 'x2'] if 'x1x2' in model_file else (
                ['x1', 'x2', 'x3']))
    test_dataset = SpectrumDataset(images, y.reindex(inference_idxs))
    test_loader = DataLoader(test_dataset, batch_size=128, 
                                 shuffle=False, num_workers = os.cpu_count())

    # print(time.time() - start)
    input_dropout = float(model_file.split('id')[-1].split('_')[0]) if 'id' in model_file else 0
    stem_dropout = float(model_file.split('sd')[-1].split('_')[0]) if 'sd' in model_file else 0
    rc_dropout = float(model_file.split('rd')[-1].split('_')[0]) if 'rd' in model_file else 0
    model = SpectrumModel().load_from_checkpoint('models/' + model_file).eval()

    # print(time.time() - start)
    yps = []
    for batch in test_loader:
        with torch.no_grad():
            yps.append(model(batch[0]).numpy())
    preds = pd.DataFrame( np.vstack(yps), index = inference_idxs,
                            columns = y.columns)
    model_preds[model_file] = preds
    if len(model_preds) % 50 == 0: print()


len(model_preds)








total = 0; ct = 0
for k, v in model_preds.items():
    total += v
    ct += 1


def logit(y): return 1 / (1 + np.exp(-y))
def getLogit(y): return -np.log( 1 / y - 1)


nn_pred = total/ct
nn_pred.columns = y.columns





sub_df = pd.read_csv('sub9.csv', index_col = 0).reindex(
    inference_idxs)


targets = y.columns
for target in targets:
    print(np.corrcoef(
        logit(0.75 * nn_pred[target]), (sub_df[target]))[0, 1])








# ### LGB Model

import lightgbm as lgb
import datetime


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, KFold


from sklearn.linear_model import LogisticRegression


from sklearn.preprocessing import StandardScaler





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








targets = y.columns
targets











model_files = [f for f in os.listdir(MODEL_PATH)
                  if 'models.pkl' in f]
len(model_files)


lgb_groups = defaultdict(list)


for file in model_files:
    lgb_groups[file.split('_rs')[0]].append(file)
           


len(lgb_groups)





print(' --- Running LGB Models ---')
lgb_preds = {};
start = time.time()
for k, v  in lgb_groups.items():
    if len(lgb_preds) > len(model_files) * 0.05:
        print('{:.0f}s of ~{:.0f}s elapsed'.format(time.time() - start,
                                             (time.time() - start)/len(lgb_preds) * len(model_files)
                                           ))

    
    RANDOM_TRIM = int(k.split('_')[2].replace('rtrim', ''))
    PREV_OFFSET = int(k.split('_')[3].replace('offset', ''))
    print('\nRunning models with rtrim {}, offset {}'.format(RANDOM_TRIM, PREV_OFFSET))

    r = Parallel(os.cpu_count())(delayed(getBasics)(sample) 
         for sample in meta[meta.split != 'train'].itertuples())

    feats = dict(zip(meta[meta.split != 'train'].sample_id, r))
    inference_idxs = meta[meta.split != 'train'].sample_id.tolist()

    feat_df = pd.DataFrame.from_dict(feats, orient = 'index', dtype = np.float32)

    temp_cols = [c for c in feat_df.columns if 'temp' in c or 'ppt' in c]
    non_temp_cols = [c for c in feat_df.columns if c not in temp_cols]
    feat_df.loc[:, temp_cols] = feat_df[temp_cols].fillna(-500)
    feat_df.loc[:, non_temp_cols] = feat_df[non_temp_cols].fillna(0)#-500)
    #.astype(np.float#isnull().sum()

    for model_idx, model_file in enumerate(v):

        epairs = pickle.load(open(os.path.join(MODEL_PATH, 
                            model_file.replace('.models.pkl', '.epairs.pkl')), 'rb'))


        models = pickle.load(
            open(os.path.join(MODEL_PATH, model_file), 'rb'))


        y_preds = {c: [] for c in targets}
        for target in targets:
            x_test, y_test = feat_df.reindex(inference_idxs), y.reindex(inference_idxs)
            element_pairs = epairs[target][0]#getElementPairs(target, x_train, y_train) 
            x_test = addElementPairFeatures( x_test, element_pairs) 

            clf = models[target][0]
            y_pred = pd.Series(clf.predict_proba(x_test)[:, 1], y_test.index)
            y_preds[target].append(y_pred)



        # collect 
        y_preds = {k: pd.concat(yp) for k, yp in y_preds.items()}
        all_y_preds = pd.DataFrame(y_preds)
        lgb_preds[model_file] = getLogit(all_y_preds)
        if len(lgb_preds) % 25 == 0: print()


total = 0; ct = 0
for v in lgb_preds.values():
    total += v
    ct += 1


lgb_pred = total / ct





sub_df = pd.read_csv('sub9.csv', index_col = 0).reindex(
    inference_idxs)


targets = y.columns
for target in targets:
    print(np.corrcoef(
        logit(0.75 * nn_pred[target]), (sub_df[target]))[0, 1])








weights = pd.read_csv('model_weights.csv', index_col = 0)
weights.sum()


plt.matshow(weights.T); plt.colorbar();





weights.index


all_preds = defaultdict(list)


for k, v in model_preds.items():
    match = [w for w in weights.index if k.startswith(w + '_')]
    assert len(match) <= 1
    if len(match) == 1:
        pass;
        all_preds[match[0]].append(v)
        # print(k, match)
    else:
        print(k)


for k, v in lgb_preds.items():
    match = [w for w in weights.index if k.startswith(w + '_')]
    assert len(match) <= 1
    if len(match) == 1:
        pass;
        all_preds[match[0]].append(v)
        # print(k, match)
    else:
        print(k)





avg_preds = {}
for k, v in all_preds.items():
    preds = pd.concat(v)
    avg = preds.groupby(preds.index).mean().astype(np.float32)
    avg_preds[k] = avg





preds = {}
all_pred_dfs = {}
for target in y.columns:
    all_pred_dfs[target] = pd.DataFrame({k: avg_preds[k][target] 
                                for k in sorted(avg_preds.keys())})

    assert all(weights.index == all_pred_dfs[target].columns)

    preds[target] = (all_pred_dfs[target] * weights[target]).sum(axis = 1)


preds = logit(pd.DataFrame(preds))


preds





# SAM samples 
sam_total = 0.7
nn_wt = 0.25
lgb_wt = sam_total - nn_wt; 


blend_df = pd.concat([v for k, v in model_preds.items()  
                       if 'x1' not in k and 'x2' not in k and
                       'lgb' not in k
                         ])
blend_df = blend_df.groupby(blend_df.index).mean()


sam_pred = logit(lgb_pred * lgb_wt + blend_df * nn_wt)





sam_ids = set(meta[meta.instrument_type.str.startswith('sam')].sample_id) & set(inference_idxs)


replace_df = pd.concat(( preds.reindex(list(set(inference_idxs) - sam_ids)),
         ( (preds + sam_pred * 2)/3).reindex(sam_ids)),).reindex(inference_idxs)











sub_df = pd.read_csv('sub9.csv', index_col = 0).reindex(inference_idxs)


for target in targets:
    print(np.corrcoef(
        replace_df[target], sub_df[target]   )[0, 1])





replace_df.to_csv('sub9b.csv')







