import pandas as pd
import numpy as np
import sys
import mlflow
from azureml.core import Workspace
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error, explained_variance_score
from sklearn.base import BaseEstimator, TransformerMixin

pd.options.mode.chained_assignment = None  # default='warn'

num_of_splits = int(sys.argv[1]) if len(sys.argv) > 1 else 5
num_of_degree = int(sys.argv[2]) if len(sys.argv) > 2 else 2
sel_of_model = sys.argv[3] if len(sys.argv) > 3 else 'LR'

discrete_props = ['Direction'] # demonstrate which column of data is discrete feature (indicating others are linear) 

def fillnan(df, col_name):
    num=0
    for i in df[col_name].notnull():
        if i is False or df[col_name][num]=='NaN':
            j=1
            while pd.isna(df[col_name][num+j]): j+=1 # check whether the next one is nan or not, if it is, skip to its next
            if col_name=='Direction': # for discrete feature(s) (here only the Direction)
                if num+j<len(df): df[col_name][num]=df[col_name][num+j] # set the next not nan value as wind direction of this one
                else: df[col_name][num]=df[col_name][num-1] # set to previous one
            else: # for all other continuous feature(s)
                if pd.isna(df[col_name][num-1]) or num-1<0: df[col_name][num]=df[col_name][num+j] # if there is no previous one, just use the next one as this one value
                else: df[col_name][num]=df[col_name][num-1]-(df[col_name][num-1]-df[col_name][num+j])/(j+1) # calculate the average of the previous value (not null) and next first not nan value to replace this nan
            #print(num,':',df[col_name][num],'\t',num+1,':',df[col_name][num+1])
        num+=1
    return df

class HourAdder(BaseEstimator,TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X["Hour"] = X.index.hour
        return X

class WindDirectCoder():
    def fit(self, X, y):
        return self

    def transform(self, X):
        encoders = {} # a dict storing all encoder of each column
        for i in X.columns: # initialize the encoder for different columns
            encoder = LabelEncoder()
            encoder.fit(X.loc[:, i])
            encoders[i] = encoder
        
        data_num = X.shape[0] # check size
        res = np.zeros(data_num).reshape(-1, 1) # initialize with 0
        for i in X.columns:
            codes = encoders[i].transform(X.loc[:, i]) if i in discrete_props else np.array(X.loc[:, i]) # check whether feature is linear or discrete
            res = np.hstack((res, codes.reshape(-1, 1))) # output
        return pd.DataFrame(res[:, 1:],columns=X.columns)

# class FillNaN():
#     def fit(self, X, y):
#         return self
    
#     def transform(self, X):
#         for i in X.columns: fillnan(X,i)
#         return X

# Preprocess the data

# Use Microsoft Azure Machine Learning Studio for experiment tracking. (pip install azureml-core)
#ws = Workspace.from_config()
#mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
#mlflow.set_tracking_uri("azureml://northeurope.api.azureml.ms/mlflow/v1.0/subscriptions/b33cbdfb-bc26-490c-af0b-0518d16378fc/resourceGroups/ITU_BDM_ResourceGroup/providers/Microsoft.MachineLearningServices/workspaces/BDMA3ML")
#mlflow.set_tracking_uri("http://training.itu.dk:5000/") # ITU public tracking server
#mlflow.set_experiment("zeli - power generation prediction experiment")

# Start a run
with mlflow.start_run():
    # Load data
    df = pd.read_json("/home/lzlniu/ITU_big_data_course/dataset.json", orient="split")
    # Handle missing data (Fill nan with fillnan function)
    dfTmean = df.groupby(pd.Grouper(freq='1H')).mean()
    #dfTmean = dfTmean[dfTmean['Lead_hours']==1] # further reduce data
    df = dfTmean.join(df['Direction'])
    df = fillnan(df, 'Total')
    df = fillnan(df, 'Direction')
    df = fillnan(df, 'Speed')
    if (sel_of_model == 'RFR'):
        pipeline = Pipeline([
            ("WindDirectCoder", WindDirectCoder()), # add this Direction feature is good for the model in general
            ("Poly", PolynomialFeatures(degree = num_of_degree)), # add polynomial features, num_of_degree is on the top, default 2
            ("RFR", RandomForestRegressor()), # use Random Forest model
        ])
    elif (sel_of_model == 'SVR'):
        pipeline = Pipeline([
            ("WindDirectCoder", WindDirectCoder()),
            ("Poly", PolynomialFeatures(degree = num_of_degree)),
            ("SVR", SVR()), # use Support-Vector Machine model
        ])
    else:
        pipeline = Pipeline([
            #("FillNaN", FillNaN()), # fill the row which its column without Speed or Direction
            #("HourAdder", HourAdder()), # commit it because bad performance
            ("WindDirectCoder", WindDirectCoder()),
            ("Poly", PolynomialFeatures(degree = num_of_degree)),
            ("LR", LinearRegression()), # use Linear Regression model
        ])
    metrics = [
            # name, func, scores(which store in [])
            ("Evar", explained_variance_score, []),
            ("Max", max_error, []),
            ("MAE", mean_absolute_error, []),
            ("MSE", mean_squared_error, []),
            ("R2", r2_score, []),
        ]
    X = df[["Speed","Direction"]]
    y = df["Total"]
    mlflow.log_param('splits', num_of_splits)
    mlflow.log_param('degree', num_of_degree)
    mlflow.log_param('model', pipeline.steps[-1][0])
    # Fit the pipeline
    for train, test in TimeSeriesSplit(num_of_splits).split(X,y):
        pipeline.fit(X.iloc[train],y.iloc[train])
        predictions = pipeline.predict(X.iloc[test])
        truth = y.iloc[test]
        #plt.plot(truth.index, truth.values, label = "Truth")
        #plt.plot(truth.index, predictions, label = "Predictions")
        #plt.legend()
        #plt.show()
        # Calculate and save the metrics for this fold
        for name, func, scores in metrics:
            score = func(truth, predictions)
            scores.append(score)
    # Log a summary of the metrics
    for name, _, scores in metrics:
        mean_score = np.mean(scores) # mean of all scores
        last_score = scores[-1] # only choose the last score
        #sum_log_score = np.log(np.sum(scores)) # sum up scores and log
        mlflow.log_metric(f"mean_{name}", mean_score)
        mlflow.log_metric(f"last_{name}", last_score)
        #mlflow.log_metric(f"sum_log_{name}", sum_log_score)

