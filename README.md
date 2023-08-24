[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

[![Banner Image](https://drivendata-public-assets.s3.amazonaws.com/nasa-mars-curiosity.jpg)](https://mars.drivendata.org/)

# Mars Spectrometry: Detect Evidence for Past Habitability

## Goal of the Competition
In this challenge, the competitors' goal was to build a model to automatically analyze mass spectrometry data collected for Mars exploration in order to help scientists in their analysis of understanding the past habitability of Mars.

Their models detect the presence of certain families of chemical compounds in data collected from performing evolved gas analysis (EGA) on a set of analog samples. The winning techniques seen in this repo may be used to help analyze data from Mars, and potentially even inform future designs for planetary mission instruments performing in-situ analysis. 

## What's in this Repository

This repository contains code from winning competitors in the [Mars Spectrometry: Detect Evidence for Past Habitability](https://www.drivendata.org/competitions/93/nasa-mars-spectrometry/) DrivenData challenge. Code for all  winning solutions are open source under the MIT License.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place | Team or User | Private Score | Summary of Model
--- | --- | ---   | ---
1 + Bonus  | dmytro | 0.092 | Represented the mass spectrogram as a 2D image (temperature vs m/z values) used as an input to CNN, RNN or transformer-based models. The diverse set of preprocessing configurations and models helped to achieve the diverse ensemble of models. This model also won the Bonus prize for its strong performance on SAM testbed data and its promise for application as judged by a panel of NASA scientists. Further details can be found in the [write-up](https://github.com/drivendataorg/mars-spectrometry/blob/main/1st%20Place/reports/DrivenData-MarsSpectrometry-BonusPrize-Documentation.pdf) in the winner's repo.
2   | \_NQ\_ | 0.116 | Feature engineering includes scaling m/z channels and area under the curve, peak value, peak width, and others. A LGBM model trained with these features ensembled with a neural network with 2 Conv1d modules, operating over temperature, followed by a linear layer across m/z channels and then a multi-target classifier gave the best performance.
3   | devnikhilmishra | 0.119 | Converted the multilabel problem into a binary classification problem. Used LightGBM k-fold ensemble model to get the initial predictions, then fed these predictions along with top 5k features to a 31 fold ensemble, catboost model  (which acted like a meta model)

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Benchmark Blog Post: [Mars Spectrometry: Detect Evidence for Past Habitability](<https://www.drivendata.co/blog/mars-spectrometry-benchmark/>)**

