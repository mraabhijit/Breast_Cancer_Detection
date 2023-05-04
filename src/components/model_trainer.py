import os
import sys

from src.utils import save_object, fit_model
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
            logging.info('Getting X_train, y_train, X_test and y_test from train and test data')
            X_train, y_train = (
                train_arr[:,:-1],
                train_arr[:,-1],
                # test_arr[:,:-1],
                # test_arr[:,-1]
                )
            
            logging.info("Defining the prediction model.")

            model_to_use = LogisticRegression()

            fit_model(X_train=X_train, y_train=y_train, model=model_to_use)
            # print(model_report)
            # print("="*40)
            # logging.info(f'Model Report: {model_report}')

            save_object(
                file_path=self.model_trainer_config.model_path,
                obj=model_to_use
            )

        except Exception as e:
            logging.info('Error occured in initiate_model_trainer')
            raise CustomException(e , sys)