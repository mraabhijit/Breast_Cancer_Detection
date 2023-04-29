import os
import sys

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

from sklearn.linear_model import LogisticRegression

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    model_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Defining the prediction model.")
            
            models = {
                'Logistic Regression': LogisticRegression()
            }

            model_to_use = models["Logistic Regression"]

            save_object(
                file_path=self.model_trainer_config.model_path,
                obj=model_to_use
            )

        except Exception as e:
            logging.info('Error occured in initiate_model_trainer')
            raise CustomException(e , sys)