import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation Started')

            numerical_cols = ['radius_mean', 'texture_mean', 'perimeter_mean',
                              'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                              'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                              'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                              'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
                              'fractal_dimension_se', 'radius_worst', 'texture_worst',
                              'perimeter_worst', 'area_worst', 'smoothness_worst',
                              'compactness_worst', 'concavity_worst', 'concave_points_worst',
                              'symmetry_worst', 'fractal_dimension_worst']

            logging.info('Numeical Pipeline Initiated')
            num_pipeline = Pipeline(
                steps = [
                ('Imputer', SimpleImputer(strategy='median')),
                ('Scaler', StandardScaler())
                ]
            )

            logging.info('Numerical Pipeline Created')


            logging.info('Column Transformer Initiated')
            preprocessor = ColumnTransformer(
                [
                ('num_pipeline', num_pipeline, numerical_cols)
                ]
            )

            logging.info('Pipeline Completed')

            return preprocessor

        except Exception as e:
            logging.info('Error occured in Data Transformation Stage')
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)


            logging.info('Read train and test data')
            logging.info(f'Train DataFrame Head: \n{train_df.head().to_string()}')
            logging.info(f"Test DataFrame Head: \n{test_df.head().to_string()}")

            logging.info("Obtaining preprocessor object")

            preprocessor_obj = self.get_data_transformation_object()

            target_column = 'diagnosis'
            columns_to_drop = ['id', target_column]

            input_feature_train_df = train_df.drop(columns = columns_to_drop, axis = 1)
            target_feature_train_df = train_df[target_column]


            input_feature_test_df = test_df.drop(columns=columns_to_drop, axis = 1)
            target_feature_test_df = test_df[target_column]

            # Transforming using preprocessor obj
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info('Train and test data transformed.')

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
                )

            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info('Error Occured in initiate_data_transformation')
            raise CustomException(e, sys)   