import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.logger.logging import logging
from src.exception.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from src.utils.utils import time_col_transform, save_object, loc_transform


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            mean_impute_cols = ["Delivery_person_Age", "Delivery_person_Ratings",
                                "equator_distance", "ordered_picked_diff", "Delivery_location_longitude",
                                'Delivery_location_latitude']
            mode_impute_cols = ["multiple_deliveries", 'Vehicle_condition']
            ordinal_encode_cols = ["City", "Type_of_vehicle", "Type_of_order", "Festival",
                                   "Weather_conditions", "Road_traffic_density"]

            # column categories
            city_columns = ["Urban", "Semi-Urban", "Metropolitian"]
            type_of_vehicle_columns = ["bicycle", "electric_scooter", "scooter", "motorcycle"]
            type_of_order_columns = ["Drinks", "Snack", "Meal", "Buffet"]
            road_traffic_density_columns = ["Low", "Medium", "High", "Jam"]
            festival_columns = ["No", "Yes"]
            weather_conditions_columns = ['Cloudy', 'Sunny', 'Windy', 'Fog', 'Sandstorms', 'Stormy']

            num_pipeline = Pipeline(steps=[
                ("num_impute", SimpleImputer(strategy="mean")),
            ])

            cat_pipeline = Pipeline(steps=[
                ("cat_impute", SimpleImputer(strategy="most_frequent")),
            ])

            encoding_pipeline = Pipeline(steps=[
                ("cat_impute", SimpleImputer(strategy="most_frequent")),
                ("ordinal_encode", OrdinalEncoder(categories=[city_columns, type_of_vehicle_columns,
                                                              type_of_order_columns,
                                                              festival_columns, weather_conditions_columns,
                                                              road_traffic_density_columns])),
            ])

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, mean_impute_cols),
                ("cat", cat_pipeline, mode_impute_cols),
                ("oc", encoding_pipeline, ordinal_encode_cols),
            ])

            logging.info("Pipeline completed")
            return preprocessor

        except Exception as error:
            logging.info("Exception occurred in pipeline creation")
            raise CustomException(error, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data transformation initiated")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')
            logging.info("Data has been successfully read")

            train_df = time_col_transform(train_df)
            train_df = loc_transform(train_df)

            test_df = time_col_transform(test_df)
            test_df = loc_transform(test_df)
            logging.info("Column transformation has been successfully completed")

            preprocessor_obj = self.get_data_transformation_object()
            logging.info("Obtained preprocessor object")
            target_name = "Time_taken (min)"
            drop_columns = [target_name, "ID"]

            input_feature_train_df = train_df.drop(drop_columns, axis=1)
            input_feature_test_df = test_df.drop(drop_columns, axis=1)

            target_feature_train = train_df["Time_taken (min)"]
            target_feature_test = test_df["Time_taken (min)"]

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test)]

            logging.info("Data transformation completed")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessor_obj)
            logging.info("Preprocessor pickle file saved")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as error:
            logging.info("Error occurred in data transformation ")
            raise CustomException(error, sys)
