from src.logger import logging
from src.exception import CustomException
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
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    except Exception as e:
        logging.info('Error occured in load_object')
        raise CustomException(e, sys)
    