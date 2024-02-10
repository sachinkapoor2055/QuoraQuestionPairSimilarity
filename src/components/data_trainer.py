import os  
import sys
import config
import matplotlib as plt
import seaborn as sns
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException  
from src.logger import logging

class ModelTrainer:
    def __init__(self, train_df, test_df, cv_df):
        try:
            self.train_df = train_df  # Training DataFrame
            self.test_df = test_df  # Testing DataFrame
            self.cv_df = cv_df  # Cross-validation DataFrame
        except Exception as e:
            logging.error("Error initializing DataTextFeaturization: DataFrame not found")
            raise CustomException("DataFrame not found", sys)
        
    def model_performance_metrics(self, y_test, y_pred):
        # Print a separator line for better visibility
        print("*" * 15 + "Performance metric" + "*" * 15)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        
        # Calculate the confusion matrix
        C = confusion_matrix(y_test, y_pred)
        
        # Calculate precision matrix by normalizing each column of the confusion matrix
        A = (((C.T) / (C.sum(axis=1))).T)
        
        # Calculate recall matrix by normalizing each row of the confusion matrix
        B = (C / C.sum(axis=0))
        
        # Define class labels
        labels = [1, 2]
        
        # Plot confusion matrix heatmap
        print("-" * 20, "Confusion matrix", "-" * 20)
        plt.figure(figsize=(3, 3))
        sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Class')
        plt.ylabel('Original Class')
        plt.show()

        # Print and plot precision matrix
        print("-" * 20, "Precision matrix (Column Sum=1)", "-" * 20)
        precision = precision_score(y_test, y_pred)
        print(f"Precision: {precision:.4f}")
        plt.figure(figsize=(3, 3))
        sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Class')
        plt.ylabel('Original Class')
        plt.show()
        
        # Print and plot recall matrix
        print("-" * 20, "Recall matrix (Row Sum=1)", "-" * 20)
        recall = recall_score(y_test, y_pred)
        print(f"Recall: {recall:.4f}")
        plt.figure(figsize=(3, 3))
        sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Class')
        plt.ylabel('Original Class')
        plt.show()
        
        # Print F1 Score
        print("-" * 20, "F1 Score", "-" * 20)
        f1 = f1_score(y_test, y_pred)
        print(f"F1 Score: {f1:.4f}")
        
        # Return calculated metrics
        return accuracy, precision, recall, f1
    
    def initiate_model_trainer(self):
        models = config.models
        hyperparameters = config.hyperparameters
        models_report = []

        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            if model_name in hyperparameters:
                param_grid = hyperparameters[model_name]
                grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_log_loss', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
            else:
                best_model = model
            
            calibrated_model = CalibratedClassifierCV(best_model, method="sigmoid", cv='prefit')
            calibrated_model.fit(X_train, y_train)
            
            # Evaluate on training set
            train_log_loss = log_loss(y_train, calibrated_model.predict_proba(X_train))
            
            # Evaluate on cross-validation set
            cv_log_loss = log_loss(y_cv, calibrated_model.predict_proba(X_cv))
            
            # Evaluate on test set
            test_log_loss = log_loss(y_test, calibrated_model.predict_proba(X_test))
            
            predict_y = calibrated_model.predict(X_test)
            knn_misclassified_points = np.count_nonzero((predict_y - y_test) != 0) / y_test.shape[0]

            # Calculate additional model performance metrics
            knn_accuracy, knn_precision, knn_recall, knn_f1 = self.model_performance_metrics(y_test, predict_y)

            # Store results in models_report
            models_report.append({
                "model_name": model_name,
                "train_loss": train_log_loss,
                "cv_log_loss": cv_log_loss,
                "test_loss": test_log_loss,
                "knn_accuracy": knn_accuracy,
                "knn_precision": knn_precision,
                "knn_recall": knn_recall,
                "knn_f1": knn_f1
            })

        ## To get best model score from dict
        sorted_models_report = sorted(models_report, key=lambda x: x['test_loss'])
        best_model = sorted_models_report[0]
                    
            