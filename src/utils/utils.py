import os
import sys
import pickle
import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.exception.exception import CustomException
from src.logger.logging import logging


def order_data_transform(X):
    X["month"] = pd.to_datetime(X["Order_Date"], format='%d-%m-%Y').dt.month
    X = X.drop(["Order_Date"], axis=1)
    return X


def time_col_transform(X):
    X = X.copy()
    time_cols = ["Time_Orderd", "Time_Order_picked"]
    for col in time_cols:
        if col == "Time_Order_picked":
            X[col] = X[col].apply(strip_time)
        find_not_in_time_format(X, col)
        X[col] = pd.to_datetime(X[col], format='%H:%M')
    X["ordered_picked_diff"] = X["Time_Order_picked"] - X["Time_Orderd"]
    X['ordered_picked_diff'] = X['ordered_picked_diff'].dt.total_seconds() // 60
    X = X.drop(time_cols, axis=1)
    return X


def find_not_in_time_format(df, col):
    li = df[col].value_counts().index
    not_time_list = []

    # Define the format string that corresponds to your time format
    format_str = '%H:%M'

    # Loop over each value in the dataset
    for i in li:
        try:
            # Try to parse the value as a time using datetime.strptime()
            datetime.datetime.strptime(i, format_str)
        except ValueError:
            not_time_list.append(i)

    for time in not_time_list:
        df[col].replace(time, np.nan, inplace=True)
    return df


def strip_time(col):
    le = col.split(":")
    if len(le) == 3:
        return f"{le[0]}:{le[1]}"
    else:
        return col


def loc_transform(df):
    df["lat"] = df["Delivery_location_latitude"] - df["Restaurant_latitude"]
    return df


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as err:
        raise CustomException(err, sys)


def load_object(file_path):
    try:
        # read the final model pickle file
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occurred in load_object function utils')
        raise CustomException(e, sys)
