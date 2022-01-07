from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import logger
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

class f_sel:
    def __init__(self,file):
        self.log_path=file
        self.log_writer=logger.App_Logger()

    def select(self,df):
        try:
            bestfeatures = SelectKBest(k=15)
            X=df.drop(columns='salary')
            Y=df.salary
            fit = bestfeatures.fit(X,Y)
            dfscores = pd.DataFrame(fit.scores_)
            dfcolumns = pd.DataFrame(X.columns)
            #concat two dataframes for better visualization
            featureScores = pd.concat([dfcolumns,dfscores],axis=1)
            featureScores.columns = ['Specs','Score']  #naming the dataframe columns
            return featureScores
        except Exception as e:
            self.log_writer.log(self.log_path, e)

    def vif_score(self,df):
        try:
            X=df.drop(columns='salary')
            X.astype(int)
            vif = pd.DataFrame()
            vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif['variable'] = X.columns
            return vif
        except Exception as e:
            self.log_writer.log(self.log_path, e)