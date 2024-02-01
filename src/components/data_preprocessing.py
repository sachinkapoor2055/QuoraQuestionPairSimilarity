import sys  
import contractions  
import string  
import nltk  
import config  
from bs4 import BeautifulSoup  
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
from nltk.stem import PorterStemmer  

from src.exception import CustomException  
from src.logger import logging  

nltk.download('stopwords')  

class DataPreprocessing:
    def __init__(self, train_df, test_df, cv_df):
        """
        Constructor method for DataPreprocessing class.

        Parameters:
            train_df (DataFrame): DataFrame for training data.
            test_df (DataFrame): DataFrame for testing data.
            cv_df (DataFrame): DataFrame for cross-validation data.
        """
        try:
            self.train_df = train_df  # Training DataFrame
            self.test_df = test_df  # Testing DataFrame
            self.cv_df = cv_df  # Cross-validation DataFrame
            print(self.train_df.shape)
            print(self.test_df.shape)
            print(self.cv_df.shape)
        except Exception as e:
            logging.error("Error initializing DataPreprocessing: DataFrame not found")
            raise CustomException("DataFrame not found", sys)

    def preprocess(self, df, columns):
        """
        Method to preprocess text data in the DataFrame.

        Parameters:
            df (DataFrame): DataFrame to preprocess.
            columns (list): List of column names containing text data.

        Returns:
            DataFrame: Processed DataFrame.
        """
        def preprocess_text(text):
            """
            Function to preprocess text data.

            Parameters:
                text (str): Text data to preprocess.

            Returns:
                str: Preprocessed text data.
            """
            # Remove HTML tags
            # print(text)
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text()
            
            # Expand contractions
            text = contractions.fix(text)
            
            # Remove punctuation
            translator = str.maketrans('', '', string.punctuation)
            text = text.translate(translator)
            
            # Tokenize and remove stopwords
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(text)
            filtered_words = [word for word in words if word.lower() not in stop_words]
            text = ' '.join(filtered_words)
            
            # Perform stemming
            porter = PorterStemmer()
            words = word_tokenize(text)
            stemmed_words = [porter.stem(word) for word in words]
            text = ' '.join(stemmed_words)
            
            return text

        # Apply preprocessing to each column containing text data
        for column in columns:
            df[column] = df[column].str.lower().apply(preprocess_text)

        return df
    
    def initiate_data_preprocessing(self):
        """
        Method to initiate the data preprocessing process.

        Returns:
            DataFrame: Preprocessed training DataFrame.
            DataFrame: Preprocessed testing DataFrame.
            DataFrame: Preprocessed cross-validation DataFrame.
        """
        # Preprocess text data in training, testing, and cross-validation DataFrames
        self.train_df = self.preprocess(self.train_df, config.categorical_columns)
        self.test_df = self.preprocess(self.test_df, config.categorical_columns)
        self.cv_df = self.preprocess(self.cv_df, config.categorical_columns)

        # Return preprocessed DataFrames
        return self.train_df, self.test_df, self.cv_df
