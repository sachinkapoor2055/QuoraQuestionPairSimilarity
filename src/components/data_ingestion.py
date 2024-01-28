import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging

class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join('artifacts', 'raw.csv')
        self.train_data_path = os.path.join('artifacts', 'train.csv')
        self.test_data_path = os.path.join('artifacts', 'test.csv')
        
    def initiate_data_ingestion(self):
        logging.info("Data ingestion task is about to start")
        try:
            df = pd.read_csv('notebook/data/train.csv')
            logging.info("Successfully loaded the dataset")

            logging.info("Train-test split has started")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
            df.to_csv(self.raw_data_path, index=False)
            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)
            logging.info("Ingestion of the data is completed")
        except FileNotFoundError:
            logging.error("Data file not found. Please check the data path.")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise CustomException(e, sys)
        
        return(self.train_data_path, self.test_data_path)


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()