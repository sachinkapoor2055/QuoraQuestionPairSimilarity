import pandas as pd
from googletrans import Translator
from src.exception import CustomException  
from src.logger import logging  
import config  

class DataCleaning:
    def __init__(self, train_df, test_df, cv_df):
        """
        Constructor method for DataCleaning class.

        Parameters:
            train_df (DataFrame): DataFrame for training data.
            test_df (DataFrame): DataFrame for testing data.
            cv_df (DataFrame): DataFrame for cross-validation data.
        """
        # Initialize DataFrames for training, testing, and cross-validation data
        self.train_df = train_df
        self.test_df = test_df
        self.cv_df = cv_df

    def handle_null_values(self, df, null_value_treatment):
        """
        Method to handle null values in the DataFrame.

        Parameters:
            df (DataFrame): DataFrame to handle null values for.
            null_value_treatment (str): Method for handling null values (drop, mean, median).
        """
        try:
            # Handling null values based on the selected treatment method
            if null_value_treatment == "drop":
                df.dropna(inplace=True)  # Drop rows with null values
            elif null_value_treatment == "mean":
                df.fillna(df.mean(), inplace=True)  # Fill null values with mean
            elif null_value_treatment == "median":
                df.fillna(df.median(), inplace=True)  # Fill null values with median
            else:
                logging.info("Invalid null value treatment option provided. Please choose from 'drop', 'mean', or 'median'.")
        except Exception as e:
            # Log error if any exception occurs during null value handling
            logging.error(f"Error handling null values: {str(e)}")
        logging.info("Successfully handled Null values")
        return df

    def translate_dataframe_columns(self, df, columns_to_translate, target_language='en'):
        """
        Method to translate text columns in the DataFrame to the specified target language.

        Parameters:
            df (DataFrame): DataFrame to translate columns for.
            columns_to_translate (list): List of column names to translate.
            target_language (str): Target language for translation (default is 'en' for English).
        """
        translator = Translator()  # Translator object for translation
        for col in columns_to_translate:
            try:
                # Translate each text value in the specified column to the target language
                df[col] = df[col].apply(lambda text: translator.translate(text, dest=target_language).text)
            except Exception as e:
                # Log error if any exception occurs during translation
                logging.error(f"Error translating column '{col}': {str(e)}")
        return df

    def initiate_data_cleaning(self):
        """
        Method to initiate the data cleaning process.

        Returns:
            DataFrame: Cleaned training DataFrame.
            DataFrame: Cleaned testing DataFrame.
            DataFrame: Cleaned cross-validation DataFrame.
        """
        try:
            # Handle null values for training, testing, and cross-validation DataFrames
            self.train_df = self.handle_null_values(self.train_df, config.null_value_treatment)
            self.test_df = self.handle_null_values(self.test_df, config.null_value_treatment)
            self.cv_df = self.handle_null_values(self.cv_df, config.null_value_treatment)

            # Uncomment below if you want to translate columns
            # columns_to_translate = config.categorical_columns
            # self.train_df = self.translate_dataframe_columns(self.train_df, columns_to_translate)
            # self.test_df = self.translate_dataframe_columns(self.test_df, columns_to_translate)
            # self.cv_df = self.translate_dataframe_columns(self.cv_df, columns_to_translate)

            # Log completion of data cleaning process
            logging.info("Cleaning of data is completed")

            # Return cleaned DataFrames
            return self.train_df, self.test_df, self.cv_df
        except Exception as e:
            # Log error if any exception occurs during data cleaning process
            logging.error(f"An error occurred during data cleaning: {e}")
            raise CustomException(e)
