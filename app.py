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
import prediction


app=Flask(__name__)


pickle_in=open('classifier.pkl','rb')

classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    log = open('logg.txt', 'w+')
    predict = prediction.predict(log)
    values=predict.get_values()
    data = predict.format_pred(values)
    m_data=[]
    for i in data.values():
        m_data.append(i)
    d=[60,40,0,1,0,0,0,1,0,0,0,0,1,0,1,0]
    pre = classifier.predict(m_data)
    if pre == [1.]:
        p = "Greator Than 50K"
    else:
        p ="Less Than 50K"
    print(pre)

    return render_template('new.html', data=p,home='index.html')



    print(data)

if __name__=="__main__":
    app.run()