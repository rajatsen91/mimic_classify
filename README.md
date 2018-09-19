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


Base Class for CI Testing. All the parameters may not be used for GAN/Regression testing
    

__X,Y,Z__: Arrays for input random variables

__max_depths__: max_depth parameter choices for xgboost e.g [6,10,13]

__n_estimators__: n_estimator parameter choices for xgboost e.g [100,200,300]

__colsample_bytrees__: colsample_bytree parameter choices for xgboost e.g [100,200,300]

__nfold__: cross validation number of folds

__train_samp__: percentage of samples to be used for training e.g -1 for default (recommended)

__nthread__: number of parallel threads for xgboost, recommended as number of processors in the machine

__max_epoch__: number of epochs when mimi function is GAN

__bsize__: batch size when mimic function is GAN or when using a deep regressor for mimifyREG

__dim_N__: dimension of noise when GAN, if None then set to dim_z + 1, can be set to a moderate value like 20

__noise__: Type of noise for regression mimic function 'Laplace' or 'Normal' or 'Mixture'

__perc__: percentage of mixture Normal for noise type 'Mixture'


__normalized__: Normalize data-set or not. Recommended setting is True for MIMIFY_REG and anything is good for GAN.  

__deep__: bool argument for mimifyREG. If true it uses a deep network for regression otherwise it uses xgb.

__deep_classifier__: if the classifier used is a deep model or xgboost. If deep model then supply this argument True. 

__params__: parameters for deep classifier. Example: {'nhid':20,'nlayers':5,'dropout':0.2} means 5 layers each with 20 neurons and train dropout of 0.2. 


The usage for both the files on synthetic data-sets can be seen in the ipython notebook named _examples_. The file `run_mimify_reg.py` gives command-line functionality to run mimify_reg from a structured folder. One such folder with datafiles in `.npy` format has been provided with the repository. An exampel to run this command line argument is provided in `example.sh`.  For mimifyGAN the same functionalities are provided as `run_mimify_GAN.py`. 


The file datagen.py in the `/src` folder has functions to generate the synthetic data-sets used in the paper. 
