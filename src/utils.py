from src.logger import logging
from src.exception import CustomException
import numpy as np
import os, sys, pickle


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    
    except Exception as e:
        logging.info('Error occured in save_object')
        raise CustomException(e, sys)
    

def fit_model(X_train, y_train, model):
    try:
        # report = {}

        logging.info(f'Training data with Logistic Regression model')
        model.fit(X_train, y_train)
        logging.info('Data trained')
        # logging.info(f"Shape of X_train: {X_train.shape}")
        # logging.info(f"Shape of X_test: {X_test.shape}")

        # logging.info(f'Predicting with Logistic Regression model')
        # # y_pred = model.predict(X_test)
        # logging.info('Prediction Complete')

        # logging.info('Evaluating scores for test data')
        # logging.info("Reshaping y_test")
        # # y_test = y_test.reshape(-1, 1)

        # logging.info("Reshaping y_pred")
        # y_pred = y_pred.reshape(-1,1)

        # test_model_score = model.score(y_test, y_pred)
        # logging.info(f'Obtained accuracy score for model')

        # report['Logistic Regression'] = round(test_model_score*100, 2)
        
        # return report
    except Exception as e:
        logging.info('Error occured in evaluate_model stage')
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    except Exception as e:
        logging.info('Error occured in load_object')
        raise CustomException(e, sys)
    