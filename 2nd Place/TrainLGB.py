#!/usr/bin/env python
# coding: utf-8

RUN = [ 'lgb']


MODEL_PATH = 'models'
PREDS_PATH = 'preds'


import random, datetime
random.seed(datetime.datetime.now().microsecond)


HELIUM = False 
SMOOTH_SERIES = False











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
            }
    

    return feats


# %%time
s = loadSample(sample)
s = cleanSample(s)
spectra = getSpectra(s)


stats = getStats(spectra)
means, totals, raw_peak, peak_temp, peak, peak_to_mean = [stats[k] for k in 
                                             ['means', 'totals', 'raw_peak', 'peak_temp', 'peak', 'peak_to_mean']]











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








sam = ( meta.set_index('sample_id').instrument_type.str.startswith('sam')
          * 1) .to_dict()








save_preds = True


sam_ids = set(meta[meta.instrument_type.str.startswith('sam')].sample_id)








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








# %%time
for j in range(20):
    RANDOM_TRIM = random.choice([ 5, 10, 20, 50, ])
    
    PREV_OFFSET = random.randrange(0, 100)

    r = Parallel(os.cpu_count())(delayed(getBasics)(sample) 
     for sample in meta[::10 if 'lgb' not in RUN else 1].itertuples())

    feats = dict(zip(meta.sample_id, r))

    feat_df = pd.DataFrame.from_dict(feats, orient = 'index', dtype = np.float32)

    temp_cols = [c for c in feat_df.columns if 'temp' in c or 'ppt' in c]
    non_temp_cols = [c for c in feat_df.columns if c not in temp_cols]
    feat_df.loc[:, temp_cols] = feat_df[temp_cols].fillna(-500)
    feat_df.loc[:, non_temp_cols] = feat_df[non_temp_cols].fillna(0)#-500)
    assert set.issubset(set(y.index), set(feat_df.index))

    for i in range(5):
        n_folds = 5
        folds_rs = datetime.datetime.now().microsecond
        folds = MultiLabelStratifiedKFold(n_folds, 
                             random_state = folds_rs,
                                     ).split(
                        None, y, verbose = 1)


        for fold_idx, (train_idxs, test_idxs) in enumerate(folds):
            if NO_SAM: 
                train_idxs = list(set(train_idxs) - set(sam_ids)); 
                print('removing SAM from training')

            start = time.time()
            inference_idxs = list( set(meta.sample_id) - set(train_idxs) ) 

            rdfs = []; 
            models = {c: [] for c in targets}
            y_preds = {c: [] for c in targets}
            epairs = {c: [] for c in targets}


            for tidx, target in enumerate(targets):
                x_train, y_train = feat_df.loc[train_idxs], y.loc[train_idxs]
                x_test, y_test = feat_df.reindex(inference_idxs), y.reindex(inference_idxs)
                element_pairs = getElementPairs(target, x_train, y_train) 
                x_train, x_test = [addElementPairFeatures( arr, element_pairs) 
                               for arr in [x_train, x_test]]

                random.seed(datetime.datetime.now().microsecond)
                model = RandomizedSearchCV(lgb.LGBMClassifier(seed = datetime.datetime.now().microsecond,
                                                                 ), lgb_params,
                                           cv = StratifiedKFold(n_splits = 4, shuffle = True,#n_repeats = 1,
                                                    random_state = datetime.datetime.now().microsecond),
                                           n_iter = 4, n_jobs = -1,
                                           scoring = 'neg_log_loss',
                                           random_state = datetime.datetime.now().microsecond)
                xt = noise(x_train)
                xt.loc[:, [c for c in feat_df.columns if any(c.endswith('_{}'.format(i)) 
                                for i in random.sample(np.arange(0, 100).tolist(), k = 5))] ] = np.nan
                xt.loc[:, random.sample(list(xt.columns), k = len(xt.columns) // 10)] = np.nan
                xt.loc[:, [c for c in feat_df.columns if c.startswith(target + '_')]] = np.nan
                model.fit(xt, y_train[target])
                clf = model.best_estimator_
                models[target].append(clf)
                epairs[target].append(element_pairs)

                rdf = pd.DataFrame(model.cv_results_).sort_values('rank_test_score').drop(columns = 'params')
                rdfs.append(rdf)

                y_pred = pd.Series(clf.predict_proba(x_test)[:, 1], y_test.index)
                y_preds[target].append(y_pred)
                print('{:.1f}s elapsed'.format(time.time() - start))


            # collect 
            y_preds = {k: pd.concat(yp) for k, yp in y_preds.items()}
            all_y_preds = pd.DataFrame(y_preds)
            print(runMetric(y[targets].reindex(set(all_y_preds.index) & set(y.index)), all_y_preds))

            file = '{}_rtrim{}_offset{}_folds{}_rs{}_fold{}'.format(
                         'lgb_base8' + ('nosam' if NO_SAM else ''), 
                             RANDOM_TRIM, PREV_OFFSET,
                             n_folds, folds_rs, fold_idx)

            if save_preds:
                os.makedirs(PREDS_PATH, exist_ok = True)
                os.makedirs(MODEL_PATH, exist_ok = True)
                pickle.dump(all_y_preds.astype(np.float32), 
                     open(os.path.join(PREDS_PATH, file + '.preds.pkl'), 'wb'))
                pickle.dump(models, 
                     open(os.path.join(MODEL_PATH, file + '.models.pkl'), 'wb'))
                pickle.dump(epairs, 
                     open(os.path.join(MODEL_PATH, file + '.epairs.pkl'), 'wb'))
            print('{:.1f}s elapsed'.format(time.time() - start)); print()







