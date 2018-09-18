# Prediction of Self Esteem in Internet-based interventions
Models of ordinal outcomes with varying levels of heterogeneity are applied in order to predict the self-esteem of individuals in Internet-based interventions. For this purpose, ecological momentary assessment data is utilized. The parameters are estimated by using Bayesian statistics (STAN) and 10-fold cross validation is applied. The models are evaluated by the RMSE, MAE, DIC, and WAIC.

## Setting and other information

* Clone this repository to your computer.
* Change the python directory to the path of the `PredictSelfEsteem` folder.
* Find the requirements in the `requierements.txt` file. 
* The folder contains multiple scripts for data preparation, application of each model, and visualization of results.
* Raw data `data`, prepared data (`dat_prepared`), and example results `example_res_hetero_stereo` are provided (prepared data is raw data already preprocessed by the `prepare_data` script and example results are based on the `hetero_stereo` model).
* Note that the provided data is no real world data. Because of confidential issues, it is not possible to provide real data; the data in this repository is therefore artifical.  
* For more accurate results, adjust the executed chains and iteration and warmup period. However, running the algorithms might take a while in order to finish.

