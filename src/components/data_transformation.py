import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import config


class DataTransformation:
    def __init__(self, train_data_path, test_data_path, cv_data_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.cv_data_path = cv_data_path

    def get_data_transformer_object(self):
        pass
    def initiate_data_transformation(self):
        train_df = pd.read_csv(self.train_data_path)
        test_df = pd.read_csv(self.test_data_path)
        cv_df = pd.read_csv(self.cv_data_path)

        train_df = train_df.dropna(subset=['question1', 'question2'])
        test_df = test_df.dropna(subset=['question1', 'question2'])
        cv_df = cv_df.dropna(subset=['question1', 'question2'])
