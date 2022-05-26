import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss
from omegaconf import OmegaConf
from tqdm import tqdm
import gc


class LGBMEnsembleModel:
  
    def __init__(self, fts, cat_cols, group_col, n_splits=15):
        """
        The function takes in a dataframe, a list of categorical columns, a group column, and a number
        of splits
        
        :param fts: the features you want to use to train the model
        :param cat_cols: list of categorical columns
        :param group_col: The column that you want to group by
        :param n_splits: number of splits to use for cross-validation, defaults to 15 (optional)
        """
      
        self.cat_cols = cat_cols
        self.cat_cols_mapper = {}
        self.fts = fts
        self.group_col = group_col
        self.n_splits=n_splits

        

    def comp_metric(self, y_true, y_pred):
        """
        The function takes in two arguments, y_true and y_pred, and returns the log loss between them
        
        :param y_true: the true labels
        :param y_pred: The predicted values
        :return: The log loss of the predictions.
        """
        return log_loss(y_true, y_pred)

    def preprocess(self, df, is_train=True):
        """
        It takes in a dataframe and a boolean value, and returns a dataframe with categorical columns
        mapped to integers
        
        :param df: The dataframe to be preprocessed
        :param is_train: This is a boolean parameter that tells the preprocessor whether it's being
        called on the training data or the test data, defaults to True (optional)
        :return: The dataframe with the categorical columns mapped to integers.
        """
           
        if self.cat_cols is None:
            self.cat_cols = []

        #### To do preprocessing on categorical columns
        for c in self.cat_cols:
            if c in df.columns:
                if is_train == True:
                    #### Create a dictionary to map each unique string value to an integer
                    self.cat_cols_mapper[c] = pd.Series(index=df[c].dropna().unique(), data=np.arange(df[c].nunique())).to_dict()

                df[c] = df[c].map(self.cat_cols_mapper[c])

        return df

    def fit(self, X, y, lgb_params=None):
        """
        It takes in the training data, the target variable, and the parameters for the LGBM model. It
        then splits the data into folds, and trains the model on each fold. It then prints the
        feature importances, the out of fold predictions, and the mean and standard deviation of the
        validation scores
        
        :param X: The training data
        :param y: The target variable
        :param lgb_params: This is the parameters for the LGBMClassifier. If you don't pass any
        parameters, the default ones will be used
        """

        ### Defining fts so that the order of features does not change during prediction time
        print(f'Total features being used are: {len(self.fts)}')
        print("\n")
        self.models = {}

        N_FOLDS = self.n_splits
        folds = StratifiedGroupKFold(N_FOLDS)
        oofs = np.zeros(len(X))

        val_scores = []
        val_idxs = []
        self.fi_df = pd.DataFrame()

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, X[self.group_col])):
        
            val_idxs.append(val_idx)
            X_trn, y_trn = X[self.fts].iloc[trn_idx], y.iloc[trn_idx]
            X_val, y_val = X[self.fts].iloc[val_idx], y.iloc[val_idx]
            
            del trn_idx
            _  = gc.collect()

            if lgb_params is None:
                lgb_params = {
                    'n_estimators': 1500,
                    'learning_rate': 0.05,
                    'colsample_bytree': 0.9,
                    'num_leaves': 311,
                    'max_depth': -1,
                    'max_bin': 64,
                    'min_child_samples': 20,
                    'reg_lambda': 1
                }

            clf = LGBMClassifier(**lgb_params)
                        
            callbacks = [lgb.early_stopping(100, verbose=0), lgb.log_evaluation(period=50, show_stdv=True)]

            clf.fit(X_trn, y_trn, eval_set=[(X_trn, y_trn), (X_val, y_val)], callbacks=callbacks)
            

            vp = clf.predict_proba(X_val)[:, 1]
            val_score = self.comp_metric(y_val, vp)
            print(f'Fold {fold_} val score: {val_score}')
            oofs[val_idx] = vp

            val_scores.append(val_score)
            self.models[f'lgbm_{fold_}'] = clf

            self.fi_df = self.fi_df.append(pd.DataFrame({'ft': self.fts, 'imp': clf.feature_importances_}))

            del X_trn, y_trn, X_val, y_val, val_idx

            _ = gc.collect()
            

        self.oofs = oofs
        fi = self.fi_df.groupby('ft')['imp'].mean().sort_values(ascending=False)
        print(f'\nTop 20 feature importances\n')
        print(fi[:20])

        oof_score = self.comp_metric(y, oofs)
        val_scores_std = np.round(np.std(val_scores), 4)
        val_scores_mean = np.round(np.mean(val_scores), 4)
        
        
        print(f'\n---Relogging each fold scores---\n')

        for i, val_idx in enumerate(val_idxs):
            score = self.comp_metric(y.iloc[val_idx], oofs[val_idx])
            print(f'Fold {i} val score: {score}')

        print(f'\nMean of val scores is {val_scores_mean} with a std of {val_scores_std}')
        print(f'OOF score: {oof_score}')


    def predict(self, X):
        """
        We first preprocess the data, then we replace the features not found in the test data with
        np.nan, then we make predictions using each model, and finally we average the predictions of all
        models to get the final predictions.
        
        :param X: The training data
        :return: The average of the predictions of all models
        """

        #### Preprocess testing data before predictions
        X = self.preprocess(X, is_train=False)

        ### Replacing features not found in test with np.nan
        not_found_fts = np.setdiff1d(self.fts, X.columns)     
        for f in not_found_fts:
            X[f] = np.nan
            
        preds_lst = []
        for model_name in tqdm(self.models.keys()):
            clf = self.models[model_name]
            preds_lst.append(clf.predict_proba(X[self.fts])[:, 1])

        #### Final Predictions is avg of the predictions of all models
        preds = np.array(preds_lst).mean(axis=0)

        return preds


class CatBoostEnsembleModel:
  
    def __init__(self, fts, cat_cols, group_col, n_splits=15):
        """
        The function takes in a dataframe, a list of categorical columns, a group column, and a number
        of splits. It then creates a dictionary of categorical columns and their respective mappers
        
        :param fts: the features you want to use to train the model
        :param cat_cols: list of categorical columns
        :param group_col: The column that you want to group by
        :param n_splits: number of splits to use for cross-validation, defaults to 15 (optional)
        """
      
        self.cat_cols = cat_cols
        self.cat_cols_mapper = {}
        self.fts = fts
        self.group_col = group_col
        self.n_splits=n_splits

        

    def comp_metric(self, y_true, y_pred):
        """
        The function takes in two arguments, y_true and y_pred, and returns the log loss between them
        
        :param y_true: the true labels
        :param y_pred: The predicted values
        :return: The log loss of the predictions.
        """
        return log_loss(y_true, y_pred)

    def preprocess(self, df, is_train=True):
        """
        It takes in a dataframe and a boolean value, and returns a dataframe with categorical columns
        mapped to integers
        
        :param df: The dataframe to be preprocessed
        :param is_train: This is a boolean parameter that tells the preprocessor whether it's being
        called on the training data or the test data, defaults to True (optional)
        :return: The dataframe with the categorical columns mapped to integers.
        """
           
        if self.cat_cols is None:
            self.cat_cols = []

        #### To do preprocessing on categorical columns
        for c in self.cat_cols:
            if c in df.columns:
                if is_train == True:
                    #### Create a dictionary to map each unique string value to an integer
                    self.cat_cols_mapper[c] = pd.Series(index=df[c].dropna().unique(), data=np.arange(df[c].nunique())).to_dict()

                df[c] = df[c].map(self.cat_cols_mapper[c])

        return df

    def fit(self, X, y, catboost_params=None):
        """
        It takes in the training data, the target variable, and the parameters for the CatBoost model.
        It then splits the data into n_splits folds, and trains a CatBoost model on each fold. It then
        returns the out-of-fold predictions, the feature importances, and the validation scores for each
        fold
        
        :param X: The training data
        :param y: The target variable
        :param catboost_params: This is a dictionary of parameters that you can pass to the
        CatBoostClassifier
        """

        ### Defining fts so that the order of features does not change during prediction time
        print(f'Total features being used are: {len(self.fts)}')
        print("\n")
        self.models = {}

        N_FOLDS = self.n_splits
        folds = StratifiedGroupKFold(N_FOLDS)
        oofs = np.zeros(len(X))

        val_scores = []
        val_idxs = []
        self.fi_df = pd.DataFrame()

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, X[self.group_col])):
        
            val_idxs.append(val_idx)
            X_trn, y_trn = X[self.fts].iloc[trn_idx], y.iloc[trn_idx]
            X_val, y_val = X[self.fts].iloc[val_idx], y.iloc[val_idx]
            
            del trn_idx
            _  = gc.collect()

            if catboost_params is None:
                catboost_params = {
                    'n_estimators': 1500,
                    'learning_rate': 0.02,
                    'rsm': 0.9,
                    'max_depth': 3,
                    'reg_lambda': 1
                }

            clf = CatBoostClassifier(**catboost_params)
                        
            clf.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=50, early_stopping_rounds=200)
            

            vp = clf.predict_proba(X_val)[:, 1]
            val_score = self.comp_metric(y_val, vp)
            print(f'Fold {fold_} val score: {val_score}')
            oofs[val_idx] = vp

            val_scores.append(val_score)
            self.models[f'catboost_{fold_}'] = clf

            self.fi_df = self.fi_df.append(pd.DataFrame({'ft': self.fts, 'imp': clf.feature_importances_}))

            del X_trn, y_trn, X_val, y_val, val_idx

            _ = gc.collect()
            

        self.oofs = oofs
        fi = self.fi_df.groupby('ft')['imp'].mean().sort_values(ascending=False)
        print(f'\nTop 20 feature importances\n')
        print(fi[:20])

        oof_score = self.comp_metric(y, oofs)
        val_scores_std = np.round(np.std(val_scores), 4)
        val_scores_mean = np.round(np.mean(val_scores), 4)
        
        
        print(f'\n---Relogging each fold scores---\n')

        for i, val_idx in enumerate(val_idxs):
            score = self.comp_metric(y.iloc[val_idx], oofs[val_idx])
            print(f'Fold {i} val score: {score}')

        print(f'\nMean of val scores is {val_scores_mean} with a std of {val_scores_std}')
        print(f'OOF score: {oof_score}')


    def predict(self, X):
        """
        We first preprocess the data, then we replace the features not found in the test data with
        np.nan, then we make predictions using each model, and finally we average the predictions of all
        models to get the final predictions.
        
        :param X: The training data
        :return: The predictions of the model.
        """

        #### Preprocess testing data before predictions
        X = self.preprocess(X, is_train=False)

        ### Replacing features not found in test with np.nan
        not_found_fts = np.setdiff1d(self.fts, X.columns)     
        for f in not_found_fts:
            X[f] = np.nan
            
        preds_lst = []
        for model_name in tqdm(self.models.keys()):
            clf = self.models[model_name]
            preds_lst.append(clf.predict_proba(X[self.fts])[:, 1])

        #### Final Predictions is avg of the predictions of all models
        preds = np.array(preds_lst).mean(axis=0)

        return preds


class MetaModel:
    def __init__(self, fts, cat_cols, group_col, n_splits_base=15, n_splits_meta=25, hyper_conf_path=None):
        """
        The function takes in a list of categorical columns, a list of features, a group column, a
        number of splits for the base model, a number of splits for the meta model, and a path to a
        hyperparameter configuration file
        
        :param fts: the features to be used in the model
        :param cat_cols: list of categorical columns
        :param group_col: The column that contains the group id
        :param n_splits_base: Number of splits for the base model, defaults to 15 (optional)
        :param n_splits_meta: Number of splits for the meta-model, defaults to 25 (optional)
        :param hyper_conf_path: path to the hyperparameter configuration file
        """
      
        self.cat_cols = cat_cols
        self.cat_cols_mapper = {}
        self.fts = fts
        self.group_col = group_col
        self.n_splits_base = n_splits_base
        self.n_splits_meta = n_splits_meta
        if hyper_conf_path is None:
            hyper_conf_path='../configs/hyperparams.yaml'
        self.hyper_conf = OmegaConf.load(hyper_conf_path)


    def add_meta_features(self, df, preds):
        """
        It takes in a dataframe and a list of predictions, and returns a dataframe with the predictions
        added as a column for each target label, and a list of the names of the columns that were added
        
        :param df: the dataframe that contains the features and labels
        :param preds: the predictions of the model
        :return: The dataframe and the meta features
        """
        df['preds'] = preds
        grp = df.groupby(['sample_id', 'target_label'])['preds'].mean().unstack('target_label')
        grp.columns = [f'target_label_{c}_preds' for c in grp.columns]
        df = pd.merge(df, grp, on='sample_id', how='left')

        meta_fts = ['preds'] + list(grp.columns)
        return df, meta_fts


    def fit(self, X, y):
        """
        We fit a LGBM model on the data, then we add meta features to the data and fit a CatBoost model
        on the new data
        
        :param X: The training data
        :param y: The target variable
        """

        self.lgbm_clf = LGBMEnsembleModel(fts=self.fts, cat_cols=self.cat_cols, group_col=self.group_col, n_splits=self.n_splits_base)
        print(f'Starting to fit base models ..................\n\n')
        self.lgbm_clf.fit(X, y, self.hyper_conf['lgb_base_params'])

        fi = self.lgbm_clf.fi_df.groupby('ft')['imp'].mean().sort_values(ascending=False)
        X, meta_fts = self.add_meta_features(X, self.lgbm_clf.oofs)

        ##### Take top 5000 features from initial model + newer features
        fts_new = (fi[:5000].index.tolist() + meta_fts)

        self.catboost_clf_meta = CatBoostEnsembleModel(fts=fts_new, cat_cols=None, group_col=self.group_col, n_splits=self.n_splits_meta)

        print(f'\n\nStarting to fit meta model ..................\n\n')
        self.catboost_clf_meta.fit(X, y, self.hyper_conf['catboost_meta_params'])
    
    def predict(self, X):
        """
        We first predict on the base models, then add the predictions as features to the original data,
        and finally predict on the meta model
        
        :param X: The input dataframe
        :return: The predictions of the meta model.
        """

        print(f'Predicting on base models ..................\n')
        preds = self.lgbm_clf.predict(X)

        print(f'\nAdding meta model features ..................')
        X, _ = self.add_meta_features(X, preds)

        print(f'\nPredicting on meta model ..................\n')
        preds_final = self.catboost_clf_meta.predict(X)

        return preds_final

    


        
    