README - Mars Spectrometry 

Download all data files to data/, unzip:
!unzip data/val_features.zip -d data/
!unzip data/test_features.zip -d data/
!unzip data/train_features.zip -d data/
!unzip data/supplemental_features.zip -d data/



Install Packages:
!pip install -r requirements.txt


Inference: ~1 hour on c6i.4xlarge
!python3 Inference.py
  -> Output: sub9b.csv


Training:

Cleanup:
!rm -r models
!rm -r preds
!rm -r model_weights.csv
!rm -r sub9.csv
!rm -r sub9b.csv


Train LGB, ~15 hours on c6i.4xlarge:
!python3 TrainLGB.py


Train NN, ~100 hours across r6i.large:
(note: script will run indefinitely)
!python3 TrainNNa.py


Train Ensemble, ~1 hour on c6i.4xlarge:
!python3 TrainEnsemble.py
  -> Output: sub9.csv
