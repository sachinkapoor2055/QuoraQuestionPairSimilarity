from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

raw_data_path = 'notebook/data/train.csv'
test_size = 0.2
cv_size = 0.2
numerical_columns = ['id', 'qid1','qid2']
categorical_columns = ['question1','question2']
null_value_treatment = "drop"
dependent_variable = ''
models = {
            "K Nearest Neighbors Classifier": KNeighborsClassifier(),
            "Multinomial Naive Bayes Classifier": MultinomialNB(),
            "Logistic Regression Classifier": SGDClassifier(loss='log_loss'),
            "SVM Classifier": SGDClassifier(loss='hinge'),
            "Decision Tree Classifier": DecisionTreeClassifier(),
            "Random Forest Classifier": RandomForestClassifier(),
            "Adaboost Classifier": AdaBoostClassifier(),
            "Gradient Boosting Classifier": GradientBoostingClassifier()
         }

hyperparameters={
        "K Nearest Neighbors Classifier": {
                                            'n_neighbors': [2,4,6,8,10,12,14,16,18,20]
                                          },
        "Multinomial Naive Bayes Classifier": {
                                            'alpha': [0.1, 0.5, 1.0, 2.0]
                                          },
        "Logistic Regression Classifier": {
                                            'alpha': [0.0001, 0.001, 0.01, 0.1]
                                          },
        "SVM Classifier": {
                            'alpha': [0.0001, 0.001, 0.01, 0.1]
                          },
        "Decision Tree Classifier": {
                                      'max_depth_values': [5, 10, 15],
                                      'min_samples_split_values': [2, 5, 10],
                                      'min_samples_leaf_values': [1, 2, 4]
                                    },
        "Random Forest Classifier": {
                                      'n_estimators_values': [50, 100, 150],
                                      'max_depth_values': [5, 10, 15],
                                      'min_samples_split_values': [2, 5, 10],
                                      'min_samples_leaf_values': [1, 2, 4]
                                    },
        "Adaboost Classifier": {
                                'n_estimators_values': [50, 100, 150],
                                'learning_rate_values': [0.01, 0.1, 1.0]
                               },
        "Gradient Boosting Classifier": {
                                            'n_estimators_list': [50, 100, 150],
                                            'learning_rate_list': [0.01, 0.1, 0.2],
                                            'max_depth_list': [3, 5, 7]
                                        },
        "XG Boost Classifier": {
                                'n_estimators_values': [50, 100, 150],
                                'learning_rate_values': [0.01, 0.1, 1.0],
                                'max_depth_values': [3, 5, 7]
                                }
        }





