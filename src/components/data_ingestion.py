import os
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception.exception import CustomException
from src.logger.logging import logging
import sys
import numpy as np


@dataclass
class DataIngestionConfig:
    train_path = os.path.join("artifacts", "train.csv")
    test_path = os.path.join("artifacts", "test.csv")
    raw_path = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("initiate_data_ingestion method starts")
        try:
            df = pd.read_csv(os.path.join("notebooks/data", "finalTrain.csv"))
            logging.info("Dataframe read successfully")

            # make artifacts dir
            os.makedirs(os.path.dirname(self.ingestion_config.raw_path), exist_ok=True)
            # raw data save
            df.to_csv(self.ingestion_config.raw_path, index=False)

            # train test data split
            train_set, test_set, = train_test_split(df, test_size=0.3, random_state=21)
            # train test data save
            logging.info("Train-test data was successfully split")

            train_set.to_csv(self.ingestion_config.train_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_path, index=False, header=True)
            logging.info("Ingestion of data completed")
            return self.ingestion_config.train_path, self.ingestion_config.test_path

        except Exception as err:
            logging.info('Exception occurred at Data Ingestion stage')
            raise CustomException(err, sys)

