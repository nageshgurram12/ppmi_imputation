# Multivariate-Time-Series-Imputation-with-Generative-Adversarial-Networks for PPMI Data
 author : Nageswara Rao
 
 tensorflow version:1.7 python:2.7
 #### Create conda environment with tensorflow version 1.7
 1) conda create -n tensorflow1.7 python=2.7
 2) conda activate tensorflow1.7
 
### The proposed method is a two-stage method. We first train GAN, then we train the input vector of the generator of GAN.
### To run the code, go to the Gan_Imputation folder:
 Execute the PPMI_main.py file, then we will get 3 folders named as "checkpoint" (the saved models), G_results (the generated samples), imputation_test_results (the imputed test dataset) and imputation_train_results (the imputed train dataset).

### Final result file location
Goto PPMIData/readImputedData.py and give train and test imputed files locations paths and execute it.
Fianl result file is in final_imputed_data.csv

### Assumptions:
1) The feature vector is fixed with length 134 (arg : --n-inputs)
2) Time feture is considered as 'TIME_FROM_BL' - index : 51
