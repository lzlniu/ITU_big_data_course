import pandas as pd
import numpy as np
import sys
import mlflow
from azureml.core import Workspace
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.base import BaseEstimator, TransformerMixin

num_of_splits = int(sys.argv[1]) if len(sys.argv) > 1 else 5
num_of_drgree = int(sys.argv[2]) if len(sys.argv) > 2 else 2

discrete_props = ['Direction'] # demonstrate which column of data is discrete feature (indicating others are linear) 

def fillnan(df, col_name):
    num=0
    for i in df[col_name].notnull():
        if i is False:
            j=1
            while pd.isna(df[col_name][num+j]): j+=1 # check whether the next one is nan or not, if it is, skip to its next
            if col_name=='Direction': # for discrete feature(s) (here only the Direction)
                df[col_name][num]=df[col_name][num+j] # set the next not nan value as wind direction of this one
            else: # for all other continuous feature(s)
                if pd.isna(df[col_name][num-1]): df[col_name][num]=df[col_name][num+j] # if there is no previous one, just use the next one as this one value
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
            encoder = preprocessing.LabelEncoder()
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
#mlflow.set_experiment("zeli - power generation prediction experiment")

# Start a run
with mlflow.start_run(run_name="splits5degree2"):
    # Load data
    df = pd.read_json("/home/lzlniu/ITU_big_data_course/dataset.json", orient="split")
    # Handle missing data (Fill nan with fillnan function)
    df=fillnan(df,'Total')
    df=fillnan(df,'Direction')
    df=fillnan(df,'Speed')
    pipeline = Pipeline([
        #("FillNaN", FillNaN()), # fill the row which its column without Speed or Direction
        #("HourAdder", HourAdder()), # commit it because bad performance
        ("WindDirectCoder", WindDirectCoder()), # add this Direction feature is good for the model in general
        ("Poly", PolynomialFeatures(degree = num_of_degree)), # add polynomial features, num_of_degree is on the top, default 2
        ("LinReg", LinearRegression()), # use Linear Regression model
        #("RFReg", RandomForestRegressor()), # use Random Forest model
        #("SVR", SVR(gamma=0.1, C=20.0)), # use Support Vector Machine model, which have the best performance
    ])
    metrics = [
            ("MAE", mean_absolute_error, []),
        ]
    X = df[["Speed","Direction"]]
    y = df["Total"]
    number_of_splits = num_of_splits # on the top, default 5
    mlflow.log_param('splits', num_of_splits)
    mlflow.log_param('degree', num_of_degree)
    mlflow.log_param('model', pipeline.steps[-1][0])
    # Fit the pipeline
    for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
        pipeline.fit(X.iloc[train],y.iloc[train])
        predictions = pipeline.predict(X.iloc[test])
        truth = y.iloc[test]
        plt.plot(truth.index, truth.values, label="Truth")
        plt.plot(truth.index, predictions , label ="Predictions")
        plt.show()
        # Calculate and save the metrics for this fold
        for name, func, scores in metrics:
            score = func(truth, predictions)
            scores.append(score)
    # Log a summary of the metrics
    for name, _, scores in metrics:
        # Note: Here we just log the mean of the scores.
        # Are there other summarizations that could be interesting?
        mean_score = sum(scores)/number_of_splits
        mlflow.log_metric(f"mean_{name}", mean_score)

# Do forecasting
# Get all future forecasts regardless of lead time
#forecasts = client.query("SELECT * FROM MetForecasts where time > now()")
#for_df = get_df(forecasts)
# Limit to only the newest source time
#newest_source_time = for_df["Source_time"].max()
#newest_forecasts = for_df.loc[for_df["Source_time"] == newest_source_time].copy()
# only use the Direction and Speed as input X
#newest_forecasts = pd.DataFrame(newest_forecasts,columns=['Direction','Speed'])
# Preprocess the forecasts and do predictions in one fell swoop
#pipeline.predict(newest_forecasts)
