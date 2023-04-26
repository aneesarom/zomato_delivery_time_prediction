import pandas as pd
import os
import sys
from src.utils.utils import load_object, order_data_transform, loc_transform, time_col_transform
from src.logger.logging import logging
from src.exception.exception import CustomException


class Predict:
    def __init__(self):
        pass

    def predict(self, df):
        try:
            df = order_data_transform(df)
            df = time_col_transform(df)
            df = loc_transform(df)
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            preprocessor_file = load_object(preprocessor_path)
            df = preprocessor_file.transform(df)
            logging.info("Transformation has been successfully transformed in prediction pipeline")
            model_path = os.path.join("artifacts", "model.pkl")
            model_file = load_object(model_path)
            prediction = model_file.predict(df)
            logging.info("Time taken has been successfully predicted in prediction pipeline")
            return prediction
        except Exception as error:
            logging.info("Exception has been occurred in prediction pipeline")
            raise CustomException(error, sys)


class CustomData:
    def __init__(self, delivery_person_age, delivery_person_ratings,
                 restaurant_latitude, restaurant_longitude, delivery_location_latitude,
                 delivery_location_longitude, order_date, time_orderd, time_order_picked,
                 weather_conditions, road_traffic_density, vehicle_condition, type_of_order,
                 type_of_vehicle, multiple_deliveries, festival, city):
        self.delivery_person_age = delivery_person_age
        self.delivery_person_ratings = delivery_person_ratings
        self.restaurant_latitude = restaurant_latitude
        self.restaurant_longitude = restaurant_longitude
        self.delivery_location_latitude = delivery_location_latitude
        self.delivery_location_longitude = delivery_location_longitude
        self.order_date = order_date
        self.time_orderd = time_orderd
        self.time_order_picked = time_order_picked
        self.weather_conditions = weather_conditions
        self.road_traffic_density = road_traffic_density
        self.vehicle_condition = vehicle_condition
        self.type_of_order = type_of_order
        self.type_of_vehicle = type_of_vehicle
        self.multiple_deliveries = multiple_deliveries
        self.festival = festival
        self.city = city

    def get_data_as_dataframe(self):
        df = pd.DataFrame({
            'Delivery_person_Age': [self.delivery_person_age],
            'Delivery_person_Ratings': [self.delivery_person_ratings],
            'Restaurant_latitude': [self.restaurant_latitude],
            'Restaurant_longitude': [self.restaurant_longitude],
            'Delivery_location_latitude': [self.delivery_location_latitude],
            'Delivery_location_longitude': [self.delivery_location_longitude],
            'Order_Date': [self.order_date],
            'Time_Orderd': [self.time_orderd],
            'Time_Order_picked': [self.time_order_picked],
            'Weather_conditions': [self.weather_conditions],
            'Road_traffic_density': [self.road_traffic_density],
            'Vehicle_condition': [self.vehicle_condition],
            'Type_of_order': [self.type_of_order],
            'Type_of_vehicle': [self.type_of_vehicle],
            'multiple_deliveries': [self.multiple_deliveries],
            'Festival': [self.festival],
            'City': [self.city],
        })
        return df


if __name__ == "__main__":
    custom_obj = CustomData(delivery_person_age=22, delivery_person_ratings=4.2,
                            restaurant_latitude=30, restaurant_longitude=78, delivery_location_latitude=30.05,
                            delivery_location_longitude=78.1, order_date="23-3-2022", time_orderd="17:55",
                            time_order_picked="6:10", weather_conditions="Fog", road_traffic_density="Jam",
                            vehicle_condition=2, type_of_vehicle="bicycle", type_of_order="Snack",
                            multiple_deliveries=0, festival="No", city="Urban")
    df = custom_obj.get_data_as_dataframe()
    predict_obj = Predict()
    predict_obj.predict(df)
