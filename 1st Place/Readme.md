The Winning Solution for the Mars Spectrometry: Detect Evidence for Past Habitability Challenge
===============================================================================================

The solution is based on the ensemble of diverse CNN, RNN and transformer-based deep models over 2d representations (m/z over temperature) of the mass spectrometry data. Data augmentation is applied during training and the simple ensemble is used, averaging the results of the different architecture models, trained on the differently preprocessed data.

Requirements
------------

Linux system (tested on Ubuntu 20.04), python 3.8 with virtualenv
NVidia GPU, tested on 2080ti with the necessary CUDA drivers installed to run pytorch.



Install dependencies, using python 3.8 with virtualenv

    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
    pip install -r requirementx.txt


Data preparation:
----------------

The training dataset is extracted to the data directory, with the next files expected:

    data
    ├── test_features
    ├── train_features
    ├── val_features
    ├── metadata.csv
    ├── train_labels.csv
    └── val_labels.csv

If necessary, modify the mars_spectrometry/config.py to specify the directory where the data is saved.



Preprocess train, validation and test data, around 2 minutes run time:

    cd mars_spectrometry
    python preprocess_data_v4.py

Generated files:

    ../data/folds_v4.csv
    ../data/features_pp_v4

Generate submission
-------------------

Run prediction using the supplied trained model weights, from ../../mars_spectrometry_submission_output/models/

    python predict.py prepare_submission

Submission is generated in the mars_spectrometry/submissions/ directory

Re-train models
------------

Most of the models are lightweight and takes a few hours to train, due to the large number of models over 4 folds it takes over a week to train all models sequentially.

    bash train_all_sequential.sh

Most of the models use only a few GB of VRAM during training and underutilize GPUS, it's ok to train two models in parallel on the same GPU with 10GB or VRAM or more.


It's faster re-train all models in parallel on multiple machines/GPUs using [ray](https://www.ray.io/):

Run cluster, for example two nodes with two GPUs each:

    ray start --head --port=6379 --num-gpus 2 --node-ip-address 10.0.0.1
    ray start --address='10.0.0.1:6379' --redis-password='<insert password as reported by ray>' --num-gpus 2
    
Train all models in parallel:

    python train_all_parallel.py

Any other approach can be used to train multiple models from the train_all_sequential.sh file in parallel.

Export selected models checkpoints from ../mars_spectrometry_submission_output/checkpoints/ to ../mars_spectrometry_submission_output/models/

    python train_cls.py export_models ""

