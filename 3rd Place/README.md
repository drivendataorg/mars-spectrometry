## 3rd place solution for NASA Mars Spectrometry (DrivenData)

https://www.drivendata.org/competitions/93/nasa-mars-spectrometry/data/

<br>
**License**: MIT

## Summary

* Firstly the problem is converted into a binary classification problem. Given a combination of a sample_features + compound_name, tell if the compound is present in the sample. For a given sample, features were the same for each compound, expect the feature telling which compound we were predicting. Total rows for training becomes = `Total_Samples x Total Compounds`

* I then used lightgbm 25 fold ensemble model to get the initial predictions, then fed these predictions along with top 5k features to an 31 fold, ensemble catboost model (which acted like a meta model). 2 stage modelling helped quite a lot in this competition, because the presence of one compound could help in the determination of others. Number of folds were high because I felt the data was quite small, so I wanted to have most of the samples in training. For a detail of modelling please see `src/modelling.py`

* Feature Engineering was done mostly per m/z for each sample_id, which was a key for a good score. For a detail of the feature engineering please see `src/preprocessing.py`

## Hardware and Time

Hardware: 64 GB Ram, 16 cores, 1 TB HDD

Training Time: 2 hours
<br>Inference Time: 6 minutes for val+test

## Software:

The code was run on python 3.7.12 with jupyter-lab

For installing other required python packages run:

`pip3 install -r requirements.txt`

## Steps to generate the final Solution

<br>

### Training:

`data`: contains all the competition files

Run `nbs/modelling_final.ipynb` to train the final model.
Final model is trained on train+val data

Trained Model: `models/model_final.joblib` is already present, so retraining is not required

<br>

### Inference:

To get the final output of the model on val+test data run
`nbs/predictions.ipynb`

The output is saved in `outputs/submission_final.csv` 

<br>

### Validation:

If you want to test the model by using only train_labels, and validation_labels as test data,
like in the first stage of the competition, you can run:
`nbs/modelling_train.ipynb`

<br>

### Sample Prediction Demo
`nbs/inference_sample.ipynb` contains a minimalistic example to do inference on a single sample_id
