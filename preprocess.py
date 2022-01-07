import logger
import pandas as pd
import numpy as np
import feature_engg
import s3
import data

class process:

    def __init__(self,file):
        self.log_path=file
        self.log_writer=logger.App_Logger()

    def cassandra(self):
        try:
            obj = data.cassandra_data(self.log_path)
            self.df = obj.cluster()
            self.df.to_csv('data.csv')
            return self.df
        except Exception as e:
            self.log_writer.log(self.log_path, e)
    
    def read_csv(self):
        try:
            self.df=self.cassandra()
            self.log_writer.log(self.log_path,'File retrived from cassandra')
            self.upload_s3('data.csv','data.csv')
            self.log_writer.log(self.log_path, "File stored in s3")

            self.log_writer.log(self.log_path,"File read")

            return self.df
        except Exception as e:
            self.log_writer.log(self.log_path,e)

    def upload_s3(self,file,key):
        try:
            bucket = s3.amazon_s3(self.log_path,'internshipadult')
            bucket.create_bucket()
            bucket.upload(file,key)
        except Exception as e:
            self.log_writer.log(self.log_path, e)

    def dummies(self,df,column,col_to_drop):
        try:
            r_dummy=pd.get_dummies(df[column])
            self.log_writer.log(self.log_path,"dummy created")
            r_dummy=self.drop_col(r_dummy,col_to_drop)
            df=df.join(r_dummy)
            df=self.drop_col(df,column)
        except Exception as e:
            self.log_writer.log(self.log_path, e)
        
        return df
            
    
            #self.log_writer.log(self.log_path,"dummy not created")

    def drop_col(self,data,column):
        try:
            data.drop(columns=column,inplace=True)
            self.log_writer.log(self.log_path, str(column)+" Dropped")
            return data
        except Exception as e:
            self.log_writer.log(self.log_path, e)



