from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    ingestion_obj = DataIngestion()
    train_path, test_path = ingestion_obj.initiate_data_ingestion()
    transformation_obj = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformation_obj.initiate_data_transformation(train_path, test_path)
    trainer_obj = ModelTrainer()
    model_list, r2_list = trainer_obj.initiate_model_training(train_arr, test_arr)

