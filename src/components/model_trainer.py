import os
import sys
from src.logger.logging import logging
from src.exception.exception import CustomException
from dataclasses import dataclass
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.utils.utils import save_object
from sklearn.ensemble import RandomForestRegressor
import lightgbm as ltb
import xgboost as xgb


@dataclass()
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            x_train, y_train, x_test, y_test = train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1:]
            logging.info("Train test split has been successfully completed")
            models = {
                "lr": LinearRegression(),
                "etr": ExtraTreesRegressor(random_state=21),
                "xgb": xgb.XGBRegressor(),
                "ltb": ltb.LGBMRegressor(random_state=21),
                "rfg": RandomForestRegressor(random_state=21)
            }

            r2_score_list = []
            trained_model_list = []

            for model in list(models.values()):
                model.fit(x_train, y_train)
                prediction = model.predict(x_test)
                accuracy = r2_score(y_test, prediction)
                trained_model_list.append(model)
                r2_score_list.append(accuracy)

            logging.info("Model creation completed")

            r2_max_value = max(r2_score_list)
            r2_max_index = r2_score_list.index(r2_max_value)
            best_model = trained_model_list[r2_max_index]
            best_model_name = list(models.keys())[r2_max_index]
            with open(os.path.join("artifacts", "evaluation.txt"), "w") as file:
                file.writelines(f"{best_model_name}: {round(r2_max_value, 2)*100}")
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info(f"{best_model_name}: {round(r2_max_value, 2)*100}")
            logging.info("Saved best model pickle file")

            return trained_model_list, r2_score_list
        except Exception as error:
            raise CustomException(error, sys)

