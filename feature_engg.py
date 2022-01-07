import logger
import pandas as pd
import numpy as np


class gain:
    def __init__(self,file):
        self.log_path=file
        self.log_writer=logger.App_Logger()
    
    def preprocess_gain(self,df):
        try:
            gain_or_loss = np.zeros(len(df))
            gain = df[df['capitalgain'] != 0].index
            loss = df[df['capitalloss'] != 0].index
            for val in gain:
                gain_or_loss[val]=1
            for val in loss:
                gain_or_loss[val]=-1
            df["gain_or_loss"]=gain_or_loss.astype(int)
            self.log_writer.log(self.log_path,"Gain column created")
            return df
        except:
            self.log_writer.log(self.log_path,"Gain column not created")

    def is_null(self,df):
        try:
            df = df.replace(' ?', np.nan)
            n=[]
            for i in df.columns:
                if df[i].isnull().sum()>0:
                    n.append(i)
            return n,df
        except Exception as e:
            self.log_writer.log(self.log_path, e)

    def freq_map(self,df,col):
        try:
            freq_map=df[col].value_counts().to_dict()
            df[col]=df[col].map(freq_map)
            return df
        except Exception as e:
            self.log_writer.log(self.log_path, e)
        
    def replace_nan(self,df,cols):
        try:
            if type(cols)==list:
                for col in cols:
                    df[col]=df[col].replace(np.nan,df[col].mode()[0])
                    self.log_writer.log(self.log_path,col)
            else:
                df[cols]=df[cols].replace(np.nan,df[cols].mode()[0])
            return df
        except Exception as e:
            self.log_writer.log(self.log_path, e)


