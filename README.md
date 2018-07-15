# mimic_classify
# Mimic &amp; Classify CI Test

__This is an implementation of the paper: https://arxiv.org/abs/1806.09708__
__Also uses the code base for the paper: https://arxiv.org/abs/1709.06138__

__Dependency__:
1. pytorch with cuda support is you have a gpu (follow the instructions on their website)
2. scikit-learn
3. CCIT (mentioned above)
4. pandas
5. numpy

_Please cite the above papers if this package is used in any publication._ 

There are two CI Testers one using CGAN as a mimic function and the other using a regression based MIMIC function. The parameters to be specified are as follows:

```
'''
Base Class for CI Testing. All the parameters may not be used for GAN/Regression testing
    X,Y,Z: Arrays for input random variables
    max_depths: max_depth parameter choices for xgboost e.g [6,10,13]
    n_estimators: n_estimator parameter choices for xgboost e.g [100,200,300]
    colsample_bytrees: colsample_bytree parameter choices for xgboost e.g [100,200,300]
    nfold: cross validation number of folds
    train_samp: percentage of samples to be used for training e.g -1 for default (recommended)
    nthread: number of parallel threads for xgboost, recommended as number of processors in the machine
    max_epoch: number of epochs when mimi function is GAN
    bsize: batch size when mimic function is GAN
    dim_N: dimension of noise when GAN, if None then set to dim_z + 1, can be set to a moderate value like 20
    noise: Type of noise for regression mimic function 'Laplace' or 'Normal' or 'Mixture'
    perc: percentage of mixture Normal for noise type 'Mixture'
    normalized: Normalize data-set or not
'''

```

The usage for both the files on synthetic data-sets can be seen in the ipython notebook named _examples_
