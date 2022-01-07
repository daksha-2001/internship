import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import roc_auc_score
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pickle
from imblearn.combine import SMOTETomek
import logger
import s3

class model:
    def __init__(self,file,X,Y):
        self.log_path=file
        self.log_writer=logger.App_Logger()
        self.X=X
        self.Y=Y

    def split_data(self):
        try:
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2, stratify=self.Y,random_state=1)
        except Exception as e:
            self.log_writer.log(self.log_path, e)
    def imbalance(self):
        try:
            os = SMOTETomek(0.90)
            self.X_train_ns,self.y_train_ns = os.fit_resample(self.X_train, self.Y_train)
            return self.X_train_ns,self.y_train_ns
        except Exception as e:
            self.log_writer.log(self.log_path, e)

    def logistic_reg(self):
        try:
            classifier = LogisticRegression()
            classifier.fit(self.X_train_ns,self.y_train_ns)
            y_pred = classifier.predict(self.X_test)
            cf_lr=confusion_matrix(self.Y_test, y_pred)
            as_lr=accuracy_score(self.Y_test, y_pred)
            f1_lr= f1_score(self.Y_test, y_pred, average='weighted')
            roc_lr=roc_auc_score(self.Y_test, y_pred)
            return cf_lr,as_lr,f1_lr,roc_lr
        except Exception as e:
            self.log_writer.log(self.log_path, e)

    def xgboost(self):
        try:
            """params = {
                'learning_rate': [0.5, 0.1, 0.01],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [100, 200, 300],
            }
            clf = GridSearchCV(CatBoostClassifier(), params, cv=5, scoring='precision', verbose=3)
            clf.fit(self.X_train_ns, self.y_train_ns)
            self.learning_rate = self.clf.best_params_['learning_rate']
            self.max_depth = self.clf.best_params_['max_depth']
            self.n_estimators = self.clf.best_params_['n_estimators']
            """
            classifier_XGB = XGBClassifier(learning_rate=0.01, max_depth=3, n_estimators=200, objective='binary:logistic')
            classifier_XGB.fit(self.X_train_ns, self.y_train_ns)
            y_pred = classifier_XGB.predict(self.X_test)
            cf_xgb = confusion_matrix(self.Y_test, y_pred)
            as_xgb = accuracy_score(self.Y_test, y_pred)
            f1_xgb = f1_score(self.Y_test, y_pred, average='weighted')
            roc_xgb = roc_auc_score(self.Y_test, y_pred)
            return cf_xgb, as_xgb, f1_xgb, roc_xgb
        except Exception as e:
            self.log_writer.log(self.log_path, e)

    def catboost(self):
        try:
            """params = {
                'learning_rate': [0.5, 0.1, 0.01],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [100, 200, 300],
                'loss_function': ['Logloss'],
                'eval_metric': ['AUC']
            }
            clf = GridSearchCV(CatBoostClassifier(), params, cv=5, scoring='precision', verbose=3)
            clf.fit(self.X_train_ns, self.y_train_ns)
            self.learning_rate = self.clf.best_params_['learning_rate']
            self.max_depth = self.clf.best_params_['max_depth']
            self.n_estimators = self.clf.best_params_['n_estimators']
            self.loss_function = self.clf.best_params_['loss_function']
            self.eval_metric = self.clf.best_params_['eval_metric']"""
            self.classifier_cb = CatBoostClassifier()
            self.classifier_cb.fit(self.X_train_ns, self.y_train_ns)
            y_pred = self.classifier_cb.predict(self.X_test)
            cf_cb = confusion_matrix(self.Y_test, y_pred)
            as_cb = accuracy_score(self.Y_test, y_pred)
            f1_cb = f1_score(self.Y_test, y_pred, average='weighted')
            roc_cb = roc_auc_score(self.Y_test, y_pred)
            return cf_cb, as_cb, f1_cb, roc_cb
        except Exception as e:
            self.log_writer.log(self.log_path, e)

    def random_forest(self):
        try:
            """params = {
                'learning_rate': [0.5, 0.1, 0.01],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [100, 200, 300],
                'loss_function': ['Logloss'],
                'eval_metric': ['AUC']
            }
            clf = GridSearchCV(CatBoostClassifier(), params, cv=5, scoring='precision', verbose=3)
            clf.fit(self.X_train_ns, self.y_train_ns)
            self.criterion = self.clf.best_params_['criterion']
            self.max_depth = self.clf.best_params_['max_depth']
            self.n_estimators = self.clf.best_params_['n_estimators']
            self.max_features = self.clf.best_params_['max_features']
            """
            self.classifier_RF = RandomForestClassifier(criterion='gini', max_depth=9, max_features='sqrt', n_estimators=200)
            self.classifier_RF.fit(self.X_train_ns, self.y_train_ns)
            y_pred = self.classifier_RF.predict(self.X_test)
            cf_rf = confusion_matrix(self.Y_test, y_pred)
            as_rf = accuracy_score(self.Y_test, y_pred)
            f1_rf = f1_score(self.Y_test, y_pred, average='weighted')
            roc_rf = roc_auc_score(self.Y_test, y_pred)
            return cf_rf, as_rf, f1_rf, roc_rf
        except Exception as e:
            self.log_writer.log(self.log_path, e)

    def best_model(self):
        try:
         self.split_data()
         self.imbalance()
         cf_lr,as_lr,f1_lr,roc_lr=self.logistic_reg()
         cf_xgb, as_xgb, f1_xgb, roc_xgb=self.xgboost()
         cf_cb, as_cb, f1_cb, roc_cb = self.catboost()
         cf_rf, as_rf, f1_rf, roc_rf = self.random_forest()
         if as_cb>as_rf or f1_cb>f1_rf:
             self.model=self.classifier_cb

         else:
             self.model=self.classifier_RF

         self.log_writer.log(self.log_path, "Logistic Regression: accuracy score- " + str(as_lr) + " " + "F1_score- " + str(f1_lr) + " " + "ROC_AUC_Score- " + str(roc_lr))
         self.log_writer.log(self.log_path, "XGBoost: accuracy score- " + str(as_xgb) + " " + "F1_score- " + str(f1_xgb) + " " + "ROC_AUC_Score- " + str(roc_xgb))
         self.log_writer.log(self.log_path, "Random Forrest: accuracy score- " + str(as_rf) + " " + "F1_score- " + str(f1_rf) + " " + "ROC_AUC_Score- " + str(roc_rf))
         self.log_writer.log(self.log_path, "Catboost: accuracy score- " + str(as_cb) + " " + "F1_score- " + str(f1_cb) + " " + "ROC_AUC_Score- " + str(roc_cb))

         pickle_out = open('classifier.pkl', "wb")
         bucket = s3.amazon_s3(self.log_path, 'internshipadult')
         pickle.dump(self.model, pickle_out)
         pickle_out.close()
        except Exception as e:
            self.log_writer.log(self.log_path, e)




