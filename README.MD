# Zomato Delivery Prediction App

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Evaluation](#evaluation)
- [Usage](#usage)

## Introduction

The Zomato Delivery Time Prediction project utilizes historical
Zomato order data to develop a machine learning model that 
predicts the delivery time for new orders. By leveraging 
features such as Delivery_person_Age, Delivery_person_Ratings, 
Road_traffic_density, Vehicle_condition, multiple_deliveries, 
Festival, City and other relevant parameters, the model aims to 
provide accurate delivery time estimates.

## Installation
```
conda create -p venv python==3.8
```
```
conda activate "folder location"
```

```
python setup.py install
```


## Data

The dataset used for training and evaluation is stored in the 
`artifacts` directory. It consists of a CSV file named 
`train.csv` and `test.csv`

## Preprocessing

Before training the model, it's essential to preprocess the data to ensure its quality and suitability for machine learning. The following preprocessing steps are performed:

1. Train-Test Split: Split the raw data into training and testing sets to evaluate the model's performance on unseen data.
2. Data Cleaning: Handle missing values
3. Feature Engineering: Extract relevant features from the raw data, such as converting categorical variables into numerical representations using encoding, transforming existing features.

To perform these preprocessing steps, refer to the `src/components/data_transformation.py` script in the repository. This script contains the necessary functions and code snippets to preprocess the raw data. Modify the script as per your requirements and execute it before training the model.

## Models

The ML model is built using the scikit-learn library and 
stored in the `artifacts` directory. Project utilizes the following machine
learning models for predicting delivery time:

- Linear Regression
- Extra Trees Regressor
- XGBoost Regressor
- LightGBM Regressor
- Random Forest Regressor

These models can be found in the `model_trainer.py` script in the repository.
The trained model is saved as `model.pkl`.

## Evaluation

To evaluate the model's performance r-squared, 
is used. The evaluation results can be found in the 
`artifacts` directory.

## Usage

1. Make sure you have the required dependencies installed by following the instructions in the [Installation](#installation) section.
2. Run the application.py
3. Open your web browser and navigate to the following address:
```commandline
http://127.0.0.1:5000/
```
This will launch the application's interface where you can interact with the Zomato Delivery Time Prediction system.
To train the model use the below route:
```commandline
http://127.0.0.1:5000/train
```
This will train the model and redirect to the home page