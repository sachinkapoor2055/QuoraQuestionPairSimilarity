import sys  
import distance  
from fuzzywuzzy import fuzz  
from nltk.corpus import stopwords  

from src.exception import CustomException  
from src.logger import logging  

class DataFeatureExtractor:
    def __init__(self, train_df, test_df, cv_df):
        """
        Constructor method for DataFeatureExtractor class.

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
            logging.error("Error initializing DataFeatureExtractor: DataFrame not found")
            raise CustomException("DataFrame not found", sys)

    def normal_feature_extraction(self, df):
        """
        Method to perform normal feature extraction on DataFrame.

        Parameters:
            df (DataFrame): DataFrame for feature extraction.

        Returns:
            DataFrame: DataFrame with normal features extracted.
        """
        # Calculate the frequency of occurrence for each question in 'qid1' and 'qid2'
        df['freq_qid1'] = df.groupby('qid1')['qid1'].transform('count')
        df['freq_qid2'] = df.groupby('qid2')['qid2'].transform('count')

        # Calculate the length (number of characters) of each question in 'question1' and 'question2'
        df['q1len'] = df['question1'].str.len()
        df['q2len'] = df['question2'].str.len()

        # Calculate the number of words in each question in 'question1' and 'question2'
        df['q1_n_words'] = df['question1'].apply(lambda row: len(row.split(" ")))
        df['q2_n_words'] = df['question2'].apply(lambda row: len(row.split(" ")))

        # Function to calculate the number of common words between two questions
        def normalized_word_common(row):
            w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
            w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
            return len(w1.intersection(w2))

        # Create a new feature 'word_common' by applying the function to each row
        df['word_common'] = df.apply(normalized_word_common, axis=1)

        # Function to calculate the total number of unique words in two questions
        def normalized_word_Total(row):
            w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
            w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
            return (len(w1) + len(w2))

        # Create a new feature 'word_Total' by applying the function to each row
        df['word_Total'] = df.apply(normalized_word_Total, axis=1)

        # Function to calculate the normalized word share between two questions
        def normalized_word_share(row):
            w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
            w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
            return 1.0 * len(w1 & w2) / (len(w1) + len(w2))

        # Create a new feature 'word_share' by applying the function to each row
        df['word_share'] = df.apply(normalized_word_share, axis=1)

        # Create a new feature 'freq_q1+q2' representing the sum of frequencies of 'qid1' and 'qid2'
        df['freq_q1+q2'] = df['freq_qid1'] + df['freq_qid2']

        # Create a new feature 'freq_q1-q2' representing the absolute difference of frequencies between 'qid1' and 'qid2'
        df['freq_q1-q2'] = abs(df['freq_qid1'] - df['freq_qid2'])

        return df
    
    def advance_feature_extraction(self, df):
        """
        Method to perform advanced feature extraction on DataFrame.

        Parameters:
            df (DataFrame): DataFrame for feature extraction.

        Returns:
            DataFrame: DataFrame with advanced features extracted.
        """
        def get_token_features(q1, q2):
            """
            Function to extract token-based features from two questions.

            Parameters:
                q1 (str): First question.
                q2 (str): Second question.

            Returns:
                list: List of token-based features.
            """
            token_features = [0.0]*10
            SAFE_DIV = 0.0001 
            stop_words = set(stopwords.words('english'))

            # Converting the Sentence into Tokens: 
            q1_tokens = q1.split()
            q2_tokens = q2.split()

            # Get the non-stopwords in Questions
            q1_words = set([word for word in q1_tokens if word not in stop_words])
            q2_words = set([word for word in q2_tokens if word not in stop_words])

            # Get the stopwords in Questions
            q1_stops = set([word for word in q1_tokens if word in stop_words])
            q2_stops = set([word for word in q2_tokens if word in stop_words])

            # Get the common non-stopwords from Question pair
            common_word_count = len(q1_words.intersection(q2_words))

            # Get the common stopwords from Question pair
            common_stop_count = len(q1_stops.intersection(q2_stops))

            # Get the common Tokens from Question pair
            common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

            token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
            token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
            token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
            token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
            token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
            token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

            # Last word of both questions is same or not
            if q1_tokens and q2_tokens:
                token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
            else:
                token_features[6] = 0  # or any default value you prefer

            # First word of both question is same or not
            if q1_tokens and q2_tokens:
                token_features[7] = int(q1_tokens[0] == q2_tokens[0])
            else:
                token_features[7] = 0  # or any default value you prefer

            # Average Token Length of both Questions
            token_features[9] = (len(q1_tokens) + len(q2_tokens))/2

            return token_features

        # get the Longest Common sub string
        def get_longest_substr_ratio(a, b):
            strs = list(distance.lcsubstrings(a, b))
            if len(strs) == 0:
                return 0
            else:
                return len(strs[0]) / (min(len(a), len(b)) + 1)

        def extract_features(df):
            """
            Function to extract advanced features from DataFrame.

            Parameters:
                df (DataFrame): DataFrame for feature extraction.

            Returns:
                DataFrame: DataFrame with advanced features extracted.
            """
            token_features = df.apply(lambda x: get_token_features(x["question1"], x["question2"]), axis=1).tolist()

            df["cwc_min"] = list(map(lambda x: x[0], token_features))
            df["cwc_max"] = list(map(lambda x: x[1], token_features))
            df["csc_min"] = list(map(lambda x: x[2], token_features))
            df["csc_max"] = list(map(lambda x: x[3], token_features))
            df["ctc_min"] = list(map(lambda x: x[4], token_features))
            df["ctc_max"] = list(map(lambda x: x[5], token_features))
            df["last_word_eq"] = list(map(lambda x: x[6], token_features))
            df["first_word_eq"] = list(map(lambda x: x[7], token_features))
            df["abs_len_diff"] = list(map(lambda x: x[8], token_features))
            df["mean_len"] = list(map(lambda x: x[9], token_features))

            # Computing Fuzzy Features and Merging with Dataset
            df["token_set_ratio"] = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
            df["token_sort_ratio"] = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
            df["fuzz_ratio"] = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
            df["fuzz_partial_ratio"] = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
            df["longest_substr_ratio"] = df.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)

            return df

        # Extract advanced features from DataFrame
        df = extract_features(df)
        return df

    def initiate_feature_extraction(self):
        """
        Method to initiate the feature extraction process.

        Returns:
            DataFrame: Extracted features from the training DataFrame.
            DataFrame: Extracted features from the testing DataFrame.
            DataFrame: Extracted features from the cross-validation DataFrame.
        """
        # Perform normal feature extraction on training, testing, and cross-validation DataFrames
        self.train_df = self.normal_feature_extraction(self.train_df)
        self.test_df = self.normal_feature_extraction(self.test_df)
        self.cv_df = self.normal_feature_extraction(self.cv_df)

        # Perform advanced feature extraction on training, testing, and cross-validation DataFrames
        self.train_df = self.advance_feature_extraction(self.train_df)
        self.test_df = self.advance_feature_extraction(self.test_df)
        self.cv_df = self.advance_feature_extraction(self.cv_df)

        # Return DataFrames with extracted features
        return self.train_df, self.test_df, self.cv_df
