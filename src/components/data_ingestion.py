import os
import sys

from src.logger import logging
from src.exception import CustomException

import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts', 'raw.csv')
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started.")
        try:
            df = pd.read_csv(os.path.join('notebooks/data','breast-cancer.csv'))
            logging.info("Read data as pandas DataFrame.")

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)

            # Label Encode target feature to get binary output
            df['diagnosis'] = LabelEncoder().fit_transform(df.diagnosis)
            logging.info('Label Encoded diagnosis feature in Dataframe')
            
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False)
            logging.info("Successfully saved raw data.")

            logging.info("Attempting to split data into training and testing datasets.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=1)

            train_set.to_csv(self.data_ingestion_config.train_data_path, header=True, index=False)
            test_set.to_csv(self.data_ingestion_config.test_data_path, header=True, index=False)
            logging.info("Successfully split into train and test datasets and saved.")

            logging.info("Data Ingestion successfully completed.")

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Error Occured in data_ingestion.initiate_data_ingestion")
            raise CustomException(e, sys)