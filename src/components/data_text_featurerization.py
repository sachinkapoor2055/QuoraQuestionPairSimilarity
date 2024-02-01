import sys  
import nltk  
import numpy as np  
import pandas as pd  
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  

from src.exception import CustomException  
from src.logger import logging  

nltk.download('stopwords')  

class DataTextFeaturization:
    def __init__(self, train_df, test_df, cv_df):
        """
        Constructor method for DataTextFeaturization class.

        Parameters:
            train_df (DataFrame): DataFrame for training data.
            test_df (DataFrame): DataFrame for testing data.
            cv_df (DataFrame): DataFrame for cross-validation data.
        """
        try:
            self.train_df = train_df  # Training DataFrame
            self.test_df = test_df  # Testing DataFrame
            self.cv_df = cv_df  # Cross-validation DataFrame
        except Exception as e:
            logging.error("Error initializing DataTextFeaturization: DataFrame not found")
            raise CustomException("DataFrame not found", sys)

    def BOW(self, bow_vectorizer, X_questions_df, X_numeric_df, fit_transform):
        """
        Method for Bag-of-Words (BOW) text featurization.

        Parameters:
            bow_vectorizer (CountVectorizer): CountVectorizer instance for BOW.
            X_questions_df (DataFrame): DataFrame containing text data.
            X_numeric_df (DataFrame): DataFrame containing numeric data.
            fit_transform (bool): Whether to fit and transform or only transform.

        Returns:
            DataFrame: DataFrame with BOW features.
        """
        # Combine question1 and question2 columns from training dataset
        X_questions = list(X_questions_df['question1']) + list(X_questions_df['question2'])
        
        # Transform the combined text data into bag-of-words representation for training data
        if fit_transform == True:
            X_q1_arr, X_q2_arr = np.vsplit(bow_vectorizer.fit_transform(X_questions).toarray(), 2)
        else:
            X_q1_arr, X_q2_arr = np.vsplit(bow_vectorizer.transform(X_questions).toarray(), 2)
        
        # Create DataFrames from the transformed arrays with original indices and feature names
        X_train_q1_df1 = pd.DataFrame(X_q1_arr, index=X_questions_df.index, columns=bow_vectorizer.get_feature_names_out())
        X_train_q1_df2 = pd.DataFrame(X_q2_arr, index=X_questions_df.index, columns=bow_vectorizer.get_feature_names_out())
        
        # Concatenate the created DataFrames along the columns
        X_train_bow_questions_df = pd.concat([X_train_q1_df1, X_train_q1_df2], axis=1)
        
        # Concatenate bag-of-words representation with numeric features for training data
        X_train_bow = pd.concat([X_numeric_df, X_train_bow_questions_df], axis=1)

        return X_train_bow

    def TFIDF(self, tfidf_vectorizer, X_questions_df, X_numeric_df, fit_transform):
        """
        Method for TF-IDF text featurization.

        Parameters:
            tfidf_vectorizer (TfidfVectorizer): TfidfVectorizer instance for TF-IDF.
            X_questions_df (DataFrame): DataFrame containing text data.
            X_numeric_df (DataFrame): DataFrame containing numeric data.
            fit_transform (bool): Whether to fit and transform or only transform.

        Returns:
            DataFrame: DataFrame with TF-IDF features.
        """
        # Combine question1 and question2 columns from training dataset
        X_questions = list(X_questions_df['question1']) + list(X_questions_df['question2'])
        
        # Transform the combined text data into TF-IDF representation for training data
        if fit_transform == True:
            X_q1_arr, X_q2_arr = np.vsplit(tfidf_vectorizer.fit_transform(X_questions).toarray(), 2)
        else:
            X_q1_arr, X_q2_arr = np.vsplit(tfidf_vectorizer.transform(X_questions).toarray(), 2)
        
        # Create DataFrames from the transformed arrays with original indices and feature names
        X_train_q1_df1 = pd.DataFrame(X_q1_arr, index=X_questions_df.index, columns=tfidf_vectorizer.get_feature_names_out())
        X_train_q1_df2 = pd.DataFrame(X_q2_arr, index=X_questions_df.index, columns=tfidf_vectorizer.get_feature_names_out())
        
        # Concatenate the created DataFrames along the columns
        X_train_bow_questions_df = pd.concat([X_train_q1_df1, X_train_q1_df2], axis=1)
        
        # Concatenate TF-IDF representation with numeric features for training data
        X_train_bow = pd.concat([X_numeric_df, X_train_bow_questions_df], axis=1)

        return X_train_bow

    def initiate_text_featurization(self):
        """
        Method to initiate text featurization process.

        Returns:
            DataFrame: Extracted features from the training DataFrame.
            DataFrame: Extracted features from the testing DataFrame.
            DataFrame: Extracted features from the cross-validation DataFrame.
        """
        # Create a CountVectorizer instance with a maximum of 1000 features
        bow_vectorizer = CountVectorizer(max_features=1000)
        
        # Create a TfidfVectorizer instance with a maximum of 1000 features
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)

        # Extract text and numeric features for training data using BOW and TF-IDF
        X_train_bow = self.BOW(bow_vectorizer, self.train_df[['question1', 'question2']], self.train_df.drop(['question1', 'question2'], axis=1), True)
        X_train_tfidf = self.TFIDF(tfidf_vectorizer, self.train_df[['question1', 'question2']], self.train_df.drop(['question1', 'question2'], axis=1), True)

        # Extract text and numeric features for testing data using BOW and TF-IDF
        X_test_bow = self.BOW(bow_vectorizer, self.test_df[['question1', 'question2']], self.test_df.drop(['question1', 'question2'], axis=1), False)
        X_test_tfidf = self.TFIDF(tfidf_vectorizer, self.test_df[['question1', 'question2']], self.test_df.drop(['question1', 'question2'], axis=1), False)

        # Extract text and numeric features for cross-validation data using BOW and TF-IDF
        X_cv_bow = self.BOW(bow_vectorizer, self.cv_df[['question1', 'question2']], self.cv_df.drop(['question1', 'question2'], axis=1), False)
        X_cv_tfidf = self.TFIDF(tfidf_vectorizer, self.cv_df[['question1', 'question2']], self.cv_df.drop(['question1', 'question2'], axis=1), False)

        return X_train_bow, X_train_tfidf, X_test_bow, X_test_tfidf, X_cv_bow, X_cv_tfidf
