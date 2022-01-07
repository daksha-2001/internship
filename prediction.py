


import pickle
import pandas as pd
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import numpy as np
import pandas_profiling as pf
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import roc_auc_score,accuracy_score
import logger

class predict:

    def __init__(self,file):
        self.log_path = file
        self.log_writer=logger.App_Logger()


    def get_values(self):
        try:
            B={}
            B["age"] = request.form.get('age')
            B["hours"] = request.form.get('hours')
            B['Black']=0
            B["gainorloss"] = request.form.get('gainorloss')
            B["marital_status"] = request.form.get('marital_status')
            B['Doctorate']=0
            B['HS-grad']=0
            B['Masters']=0
            B['higher-edu']=0
            B['school']=0
            B['Gov']=0
            B['Self-emp']=0
            B["gender"] = request.form.get('gender')
            B['Mexico']=0
            B['US'] = 0
            B['NotFamily'] = 0
            B["race"] = request.form.get('race')
            B["country"] = request.form.get('country')
            B["child"] = request.form.get('child')
            B["education"] = request.form.get('Education')
            B["work"] = request.form.get('workclass')
            return B
        except Exception as e:
            self.log_writer.log(self.log_path, e)

    def format_pred(self,B):
        try:
            if B['race'] == 'black':
                B['Black'] = 1
            else:
                B['Black'] = 0
            B.pop('race')
            if B['country'] == 'Mexico':
                B['Mexico'] = 1
            if B['country'] == 'United States':
                B['United States'] = 1
            B.pop('country')
            if B['child']=='yes':
                B['NotFamily']=0
            else:
                B['NotFamily']=1
            B.pop('child')
            if B['work']=='Gov':
                B['Gov']=1
            elif B['work']=='Self-emp':
                B['Self-emp']=1
            B.pop('work')
            if B['education']=='Doctorate':
                B['Doctorate']=1
            if B['education']=='Masters':
                B['Masters']=1
            if B['education']=='school':
                B['school']=1
            if B['education']=='higher-edu':
                B['higher-edu']=1
            if B['education'] == 'HS-grad':
                B['HS-grad'] = 1
            B.pop('education')
            return B
        except Exception as e:
            self.log_writer.log(self.log_path, e)





