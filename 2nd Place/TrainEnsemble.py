#!/usr/bin/env python
# coding: utf-8

# !jupyter nbconvert --no-prompt --to script 'Inference.ipynb' 








RUN = [ 'stacker']


MODEL_PATH = 'models'
PREDS_PATH = 'preds'








import random, datetime
random.seed(datetime.datetime.now().microsecond)








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














from collections import defaultdict





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





sam_ids = set(meta[meta.instrument_type.str.startswith('sam')].sample_id)











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





sam_ids = set(meta[meta.instrument_type.str.startswith('sam')].sample_id)








# for e in range(10, 101, 10):
#     # if e == 70: continue;
    
#     print([os.remove(PREDS_PATH + '/' + f)
#       for f in os.listdir(PREDS_PATH) if f.endswith('epoch{}'.format(e))])
#     print([os.remove(MODEL_PATH + '/' + f)
#        for f in os.listdir(MODEL_PATH) if 'epoch={:03d}'.format(e - 1) in f])








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
        # f
#         (f.split('x3p')[0] + '_temp' + f.split('_temp')[-1])
#             if 'x2' in f else ''.join(f.split('x3'))  
         
#         f.split('temp25_o')[0] + 'temp25_' + '-'.join(f.split('-')[2:]) if 'nn' in f else f
        
        
              
#              ).split('_rs')[0] 
#             + ( #( '_epoch' + f.split('_epoch')[-1])
#                  'e'
#                if 'epoch' in f else '')
          '_'.join(f.split('_')[:2 if 'nn' in f else 3]))
        , f) for f in files
                        
            ]
    for file in files:
        groups[file[0]].append(file[1]);
    return groups


# %%time
lgb_files = [f for f in files if 'lgb' in f and 'stack' not in f]
lgb_groups = groupFiles(lgb_files)





cleanup_lgb_groups = {k: v 
                 for k, v in lgb_groups.items() if len(v) < 5 * 5}
len(cleanup_lgb_groups)


for k, v in cleanup_lgb_groups.items():
    for e in v:
        try:
            os.remove(PREDS_PATH + '/' + e)
            os.remove(MODEL_PATH + '/' + e.replace('.preds', '.models'))
            os.remove(MODEL_PATH + '/' + e.replace('.preds', '.epairs'))
            print()
        except: 
            pass


# [f for f in os.listdir('models') if 'lgb' in f]








# lgb_groups = {k: v 
#                  for k, v in lgb_groups.items() if len(v) >= 4 * 5}
# print(len(lgb_groups), 'groups of LightGBM models')





lgb_preds = Parallel(os.cpu_count())(delayed(loadFiles)(files)
                        for files in lgb_groups.values())
lgb_preds = dict(zip(lgb_groups.keys(), lgb_preds))


sorted([(k, len(v)) for k, v in lgb_groups.items()],
           key = lambda x: -x[-1])





# files = os.listdir('preds')


# len([os.remove('preds/' + f)
#                 for f in files if 'offset88' in f 
#         and ('rs3' in f or 'rs7' in f)
#                ])








def logit(y): return 1 / (1 + np.exp(-y))
def getLogit(y): return -np.log( 1 / y - 1)

lgb_preds = {k: getLogit(v) for k, v in lgb_preds.items()}


len(lgb_preds)


# plt.matshow(list(lgb_preds.values())[0].corr())
# plt.colorbar()








nn_files = [f for f in files if 'nn_base8a' in f #and 'epoch70' in f
            and 'epoch059' in f
               # and ( 'rs2' in f or  'rs3' in f)
           ]
nn_groups = groupFiles(nn_files)
len(nn_groups)


# nn_groups = {k: v #.replace('-0,0,0', ''): v 
#                  for k, v in nn_groups.items() if len(v) >= 4 * 5}
# len(nn_groups)


nn_groups = dict(#sorted(
            [(k, v) for k, v in nn_groups.items()],
           # key = lambda x: -len(x[-1]))

)


sorted([(k, len(v)) for k, v in nn_groups.items()],
           key = lambda x: -x[-1])











# %%time
nn_preds = Parallel(os.cpu_count() * 2)(delayed(loadFiles)(files)
                        for files in nn_groups.values())


nn_preds = dict(zip(nn_groups.keys(), nn_preds))


len(nn_preds)





all_preds = {}
all_preds.update(lgb_preds)
all_preds.update(nn_preds)





all_preds = {}
all_preds.update({k: v 
                  for k, v in lgb_preds.items()
                 })
all_preds.update({k: v 
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


sorted(
    [(runMetric(y, logit(preds)), k) for k, preds in all_preds.items()],
    key = lambda x: x[-1])


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


nns = pd.concat(nn_preds.values())
nn_blend = nns.groupby(nns.index).mean()


# runMetric(y, logit(lgb_blend))


blend_df = pd.concat(all_preds.values())
blend_df = blend_df.groupby(blend_df.index).mean()


runMetric(y, logit(nn_blend * 0.6 + lgb_blend * 0.15))


runMetric(y, logit(nn_blend * 0.6 + lgb_blend * 0.2))








# np.mean(['lgb' in k for k in all_preds.keys()])


# plt.matshow(blend_df.corr())
# plt.colorbar()








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
                'l1_ratio': [0.0, 0.01, 0.03, 0.1, #0.3, 0.5,   
                            ]
            }








# %%time
rdfs = []# {target: [] for target in y.columns}
y_preds = {target: [] for target in y.columns}
models = {target: [] for target in y.columns}
all_wts = {target: [] for target in y.columns}
for target, pred_df in all_pred_dfs.items():
    print(target); 
    # pred_df = logit(pred_df)# + np.random.normal(0, 0.1, size = pred_df.shape)#.iloc[:, :: 3]
    # pred_df = logit(pred_df)
    for bag_idx in range(6):
        n_folds = 5 # random.randrange(4, 6)
        folds_rs = datetime.datetime.now().microsecond
        folds = MultiLabelStratifiedKFold(n_folds, 
                             random_state = folds_rs,
                        full = random.choice([ True, False])
                                     ).split(
                        None, 
            y, #y.reindex(list(set(meta[meta.instrument_type == 'commercial'].sample_id) & set(y.index)))
            verbose = 0)


        for fold_idx, (train_idxs, test_idxs) in enumerate(folds):
            start = time.time()
            inference_idxs = list( set(meta.sample_id) - set(train_idxs) ) 

            random.seed(datetime.datetime.now().microsecond)
                
            model = RandomizedSearchCV(
                CLR(), lr_params,
                    cv = StratifiedKFold(
                              n_splits = random.randrange(4, 6),
                                # n_repeats = random.choice([1, 2]),
                                shuffle = True,
                                random_state = datetime.datetime.now().microsecond
                        ),
                                    n_iter = random.randrange(3, 5),#choice([3, 4]),
                                    n_jobs = -1,
                            scoring = 'neg_log_loss',
                            random_state = datetime.datetime.now().microsecond)
            pdf = pred_df.drop(columns = random.sample(list(pred_df.columns),
                           k = int( (0.4 + 0.2 * random.random())
                                       * len(pred_df.columns)) ) )
            # early_cols =  [c for c in pdf.columns if 'epoch10' in c
            #                               or 'epoch20' in c or 'epoch30' in c]
            # pdf = pdf.drop(columns = random.sample(early_cols,
            #                    k = int(0.7 * len(early_cols))))
            
            
            model.fit(pdf.loc[train_idxs], 
                      y.loc[train_idxs, target])
            clf = model.best_estimator_
            # break;
            if bag_idx == 0: print(clf)
            models[target].append(clf)
            all_wts[target].append(pd.Series(clf.wts, pdf.columns))

            rdf = pd.DataFrame(model.cv_results_
                        ).sort_values('rank_test_score').drop(
                                    columns = 'params')
            rdfs.append(rdf)

            y_pred = pd.Series(clf.predict(
                        pdf.loc[inference_idxs]),#[:, 1], 
                                   inference_idxs)
            y_pred
            y_preds[target].append(y_pred)
        # break;
    # break;
        # display(rdf)
    print()


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
# 0.1001# 0.0993 -- -r1






model_weights = {}
for k, v in all_wts.items():
    df = pd.concat(v)
    df = df.groupby(df.index).sum().reindex(pred_df.columns) / len(v)
    model_weights[k] = df #/ df.sum()
model_weights =  pd.DataFrame(model_weights, index = pred_df.columns)


[c.startswith('ox') or c.endswith('silicate') for c in y.columns]


plt.matshow(model_weights.T)
plt.colorbar()


model_weights.sum()





model_weights.mean(axis = 1).sort_index().round(3).sort_values()[::-1][:30]


model_weights.shape


model_weights.mean(axis = 1).sort_values()#.to_csv('model_weights.


model_weights.to_csv('model_weights.csv')








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


lgbs = pd.concat([v for k, v in lgb_preds.items() #if 'nosam' in k
                 ])
lgb_blend = lgbs.groupby(lgbs.index).mean()


blend_df = pd.concat([v for k, v in all_preds.items()  
                       if 'x1' not in k and 'x2' not in k and
                       'lgb' not in k
                         ])
blend_df = blend_df.groupby(blend_df.index).mean()


# SAM samples (!!!)
sam_total = 0.7
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


replace_df = pd.concat(( y_pred_df
                        
                        .reindex(set(meta.sample_id) - sam_ids),
           logit( ( getLogit( (y_pred_df + sam_pred_df * 2)/3)
                   # + getLogit(y_pred_df.mean()).values[None, :]
                    ) #/ 2
                  )
                        .reindex(sam_ids)),)





# y_pred_df.mean().plot()
# y_pred_df.reindex(sam_ids).mean().plot()
# replace_df.reindex(sam_ids).mean().plot()
# sam_pred_df.reindex(sam_ids).mean().plot()





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

# 0.1068



y_pred_sub.to_csv('sub9.csv', float_format = '%.5f')


# y_pred_sub.to_csv('sub8e.csv', float_format = '%.5f')








y_pred_sub.reindex(val_labels.index)


val_labels.index





[f for f in sorted(os.listdir('../')) if f.startswith('sub')]


[f for f in  sorted(os.listdir()) if f.startswith('sub')]


subs = [#pd.read_csv('../'+ f, index_col = 0)
         #   for f in sorted(os.listdir('../')) if f.startswith('sub')
        #]+ [
        pd.read_csv(f, index_col = 0)
            for f in sorted(os.listdir()) if f.startswith('sub')]


for col in y.columns:
    plt.matshow(pd.concat([s[col] for s in subs], axis = 1).corr())
    plt.title(col); plt.colorbar()










