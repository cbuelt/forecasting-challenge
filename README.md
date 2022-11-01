## Table of contents
* [General info](#general-info) 
* [Files](#files)
* [Models & Data](#models--data)
* [Technologies](#technologies)

## General info
This project contains my code for the Probabilistic Time Series Forecasting Challenge carried out by the chair of Applied Econometrics at the Karlsruher Institute of Technology (KIT). The aim of the challenge is to develop models for probabilistic forecasting for different time series datasets.

## Models & Data
### DAX forecasting
Aim: Forecasting different quantiles of the log return of the German DAX index. 

Model: WaveNet model with custom pinball loss functions.

### Temperature forecasting
Aim: Forecasting different quantiles of the hourly temperature measured in Berlin.

Model: Feed forward neural network with custom pinball loss functions.

### Wind speed forecasting
Aim: Forecasting different quantiles of the hourly wind speed measured in Berlin.

Model: Feed forward neural network with custom pinball loss functions.


## Files
| Folder | Description |
| ---- | ----------- | 
| data | Contains the data used for the models. |
| evaluation | Contains evaluation of the models and the challenge performance. |
| models | Contains all model implementations and applications. |
	
## Technologies
The project uses the following framework versions:
* Python version: 3.8.8
* TensorFlow version: 2.4.1
