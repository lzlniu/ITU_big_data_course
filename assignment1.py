## Getting the data

from influxdb import InfluxDBClient # install via "pip install influxdb "
import pandas as pd
import numpy as np

client = InfluxDBClient(host='influxus.itu.dk', port=8086 , username='lsda', password='icanonlyread')
client.switch_database('orkney')

def get_df(results):
    values = results.raw["series"][0]["values"]
    columns = results.raw["series"][0]["columns"]
    df = pd.DataFrame(values, columns = columns).set_index("time")
    df.index = pd.to_datetime(df.index) # Convert to datetime - index
    return df

# Get the power generation data from 2020-10-01 to 2021-10-01 and calculate its average for each 3 hour time period.
generation = client.query("SELECT time,mean(Total) FROM Generation where time > '2020-10-01' and time <= '2021-10-01' GROUP BY time(3h)")

# Get the weather forecasts from 2020-10-01 to 2021-10-01 with the shortest lead time (Lead hours is 1)
wind = client.query("SELECT time,Direction,Speed FROM MetForecasts where time > '2020-10-01' and time <= '2021-10-01' and Lead_hours = '1'") # earlist time '2019-03-29'

gen_df = get_df(generation)
wind_df = get_df(wind)

print('Original gen_df data num:',len(gen_df))
print('Original wind_df data num:',len(wind_df))
if (len(gen_df)>len(wind_df)): df = gen_df.join(wind_df) # left join, gen_df
else: df = wind_df.join(gen_df)
print('Combined(joined) data num:',len(df))

# df = df[df['mean'].notnull()] # drop the column without mean 
# print('Removed data with power generation column is null, remained data num:',len(df))

# df = df[df['Speed'].notnull()]
# print('removed data with wind speed column is null, remained data num:',len(df))

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

# Fill all nan
df=fillnan(df,'mean')
df=fillnan(df,'Direction')
df=fillnan(df,'Speed')

## Preprocess the data

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin

discrete_props = ['Direction'] # demonstrate which column of data is discrete feature (indicating others are linear) 

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

pipeline = Pipeline([
    #("FillNaN", FillNaN()), # fill the row which its column without Speed or Direction
    #("HourAdder", HourAdder()), # commit it because bad performance
    ("WindDirectCoder", WindDirectCoder()), # add this Direction feature is good for the model in general
    #("LinReg", LinearRegression()), # use Linear Regression model
    #("RFReg", RandomForestRegressor()), # use Random Forest model
    ("SVR", SVR(gamma=0.1, C=20.0)), # use Support Vector Machine model, which have the best performance
])

# Fit the pipeline
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(pd.DataFrame(df,columns=['Direction','Speed']), df['mean'], test_size=0.2, random_state=42)
pipeline.fit(x_train,y_train)
y_hat = pipeline.predict(x_test)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_hat)

print("Training MAE:", mae)

## Do forecasting
# Get all future forecasts regardless of lead time
forecasts = client.query("SELECT * FROM MetForecasts where time > now()")
for_df = get_df(forecasts)
# Limit to only the newest source time
newest_source_time = for_df["Source_time"].max()
newest_forecasts = for_df.loc[for_df["Source_time"] == newest_source_time].copy()
# only use the Direction and Speed as input X
newest_forecasts = pd.DataFrame(newest_forecasts,columns=['Direction','Speed'])
# Preprocess the forecasts and do predictions in one fell swoop
pipeline.predict(newest_forecasts)
