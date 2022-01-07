import preprocess
import feature_engg
import logger
import numpy as np
import feature_selection
import models
import prediction
import data

class main:
    def __init__(self,file):
        self.log_path=file
        self.log_writer=logger.App_Logger()
        
    def form_dataset(self):
        file=preprocess.process(log)
        df=file.read_csv()
        print(df.columns)
        self.gain=feature_engg.gain(log)
        cols=[' Amer-Indian-Eskimo',' Other']
        df=file.dummies(df,'race',cols)
        df=self.gain.preprocess_gain(df)
        print(df.columns)
        df=file.drop_col(df,["capitalgain","capitalloss",'fnlwgt','educationnum','occupation','id'])
        #education-num value_counts for each category is same as categories of education 
        #hence education dropped.
        null_values,df=self.gain.is_null(df)
        df=self.gain.replace_nan(df,null_values)
        df["maritalstatus"].replace(to_replace=df['maritalstatus'].unique(),value = ['single','married','single','single','single','married','single'], inplace=True)
        df=file.dummies(df,'maritalstatus','single')
        df["education"].replace(to_replace=df['education'].unique(),value = ['Bachelors','HS-grad','school','Masters','school','higher-edu','higher-edu','higher-edu','school','Doctorate','higher-edu','school','school','school','school','school'], inplace=True)
        df=file.dummies(df,'education','Bachelors')
        df['workclass'].replace(to_replace=df['workclass'].unique(),value = ['Gov','Self-emp','Private','Gov','Gov','Self-emp','Without-pay/Never-Worked','Without-pay/Never-Worked'], inplace=True)
        df=file.dummies(df,'workclass','Without-pay/Never-Worked')
        df=file.dummies(df,'salary',' >50K')
        df=file.dummies(df,'sex',' Male')
        cols=[' Cambodia', ' Canada', ' China', ' Columbia', 
        ' Cuba',' Dominican-Republic', ' Ecuador', ' El-Salvador', ' England',
        ' France', ' Germany', ' Greece', ' Guatemala', ' Haiti',
        ' Holand-Netherlands', ' Honduras', ' Hong', ' Hungary', ' India',
        ' Iran', ' Ireland', ' Italy', ' Jamaica', ' Japan', ' Laos',
        ' Nicaragua', ' Outlying-US(Guam-USVI-etc)', ' Peru', ' Philippines',
        ' Poland', ' Portugal', ' Puerto-Rico', ' Scotland', ' South',
        ' Taiwan', ' Thailand', ' Trinadad&Tobago',
        ' Vietnam', ' Yugoslavia']
        #value_counts() of above countries are less than 500 hence dropped
        df=file.dummies(df,'country',cols)
        df=file.dummies(df,'relationship',' Husband')
        df.rename(columns={' <=50K':'salary',' Female':'Gender','married':'Marital_status'},inplace=True)
        print(df.columns)
        df=file.drop_col(df,[" Unmarried","Private",' White',' Own-child',' Asian-Pac-Islander',' Wife']) 
        self.f_sel=feature_selection.f_sel(log)
        featureScores=self.f_sel.select(df)
        vif=self.f_sel.vif_score(df)
        df = df.drop(columns=' Other-relative')
        X=df.drop(columns='salary')
        Y=df['salary']
        X.to_csv('X_csv')
        Y.to_csv('Y_csv')
        file.upload_s3('X_csv','X')
        file.upload_s3('Y_csv', 'Y')
        print("done")
        print(X.columns)
        print(X,Y)
        model=models.model(log,X,Y)
        model.best_model()
        file.upload_s3('classifier.pkl', 'model(pkl file)')





log=open('logg.txt','w+')
main_obj=main(log)
main_obj.form_dataset()