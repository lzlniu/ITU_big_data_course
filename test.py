import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

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

df = pd.read_json("/home/lzlniu/ITU_big_data_course/dataset.json", orient="split")
dfTmean = df.groupby(pd.Grouper(freq='1H')).mean()
#dfTmean = dfTmean[dfTmean['Lead_hours']==1]
df = dfTmean.join(df['Direction'])
df = df[['Total','Direction','Speed']]
print(dfTmean)
print(df)

df = fillnan(df, 'Total')
df = fillnan(df, 'Direction')
df = fillnan(df, 'Speed')

print(df)

count=0
count_nan=0
for i in df['Total'].notnull():
    if i<0:
        count_nan+=1
        print(df.iloc[count,:])
    count+=1

print(count_nan,count)


