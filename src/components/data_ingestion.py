import os  
import sys  
import config  
import pandas as pd  
from sklearn.model_selection import train_test_split  

from src.exception import CustomException  
from src.logger import logging  

from data_cleaning import DataCleaning  
from data_feature_extractor import DataFeatureExtractor  
from data_preprocessing import DataPreprocessing  
from data_text_featurerization import DataTextFeaturization  

class DataIngestion:
    def __init__(self):
        """
        Constructor method to initialize paths for raw and processed data.
        """
        self.raw_data_path = os.path.join('artifacts', 'raw.csv')  # Path to raw data
        self.train_data_path = os.path.join('artifacts', 'train.csv')  # Path to training data
        self.test_data_path = os.path.join('artifacts', 'test.csv')  # Path to testing data
        self.cv_data_path = os.path.join('artifacts', 'cv.csv')  # Path to cross-validation data
        
    def initiate_data_ingestion(self):
        """
        Method to ingest raw data, split it into train, test, and cv sets, and save them as CSV files.
        
        Returns:
            DataFrame: Train data
            DataFrame: Test data
            DataFrame: Cross-validation data
        """
        try:
            # Read raw data from configured path
            df = pd.read_csv(config.raw_data_path)
            logging.info("Successfully loaded the raw dataset")

            # Split data into train, test, and cv sets
            train_df, test_df = train_test_split(df, test_size=config.test_size, random_state=42)
            train_df, cv_df = train_test_split(train_df, test_size=config.cv_size, random_state=42)
            logging.info("Train-test-cv split completed")

            # Ensure artifacts directory exists
            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)

            # Save dataframes to CSV files
            train_df.to_csv(self.train_data_path, index=False)
            test_df.to_csv(self.test_data_path, index=False)
            cv_df.to_csv(self.cv_data_path, index=False)
            logging.info("Ingestion of the data is completed")

        except FileNotFoundError as e:
            # Log error if raw data file is not found
            logging.error(f"Raw data file not found: {config.raw_data_path}")
            raise CustomException(e, sys)
        except Exception as e:
            # Log error if any other exception occurs
            logging.error(f"An error occurred: {e}")
            raise CustomException(e, sys)
        
        # Return train, test, and cv DataFrames
        return train_df, test_df, cv_df 

if __name__=="__main__":
    # Instantiate DataIngestion class
    DI = DataIngestion()
    train_df, test_df, cv_df = DI.initiate_data_ingestion()
    
    # Perform data cleaning
    DC = DataCleaning(train_df, test_df, cv_df)
    train_df, test_df, cv_df = DC.initiate_data_cleaning()

    # Perform data preprocessing
    DP = DataPreprocessing(train_df, test_df, cv_df)
    train_df, test_df, cv_df = DP.initiate_data_preprocessing()

    # Perform data feature extraction
    DFE = DataFeatureExtractor(train_df, test_df, cv_df)
    train_df, test_df, cv_df = DFE.initiate_feature_extraction()

    # Perform text featurization
    DTF = DataTextFeaturization(train_df, test_df, cv_df)
    X_train_bow, X_train_tfidf, X_test_bow, X_test_tfidf, X_cv_bow, X_cv_tfidf = DTF.initiate_text_featurization()
