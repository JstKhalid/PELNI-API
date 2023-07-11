import numpy as np
import pandas as pd
import holidays
from datetime import datetime, date, timedelta
import hijri_converter
from hijri_converter import convert
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import xgboost as xgb
from fastapi import APIRouter, Depends, HTTPException, status, Response
import schemas
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import json
from sklearn.metrics import r2_score
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)



def create_df(datafile,targ_col,date_col):
    #Function dasar buat kasus pelni, isinya read, convert type sama rename
    df = pd.read_excel(datafile) #*read by excel, kemungkinan bakal di ganti readnya by list/sql karena bakal ngambil dari database
    df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d') #convert kolom timeframe ke  dateframe format
    df = df.rename(columns={date_col: 'date',targ_col: 'y'}) #rename kolom timeframe dan target prediksi
    df = df.sort_values('date')
    return df

def add_route(df,org_col,des_col):
    #Function buat nambahin + encoding rute / O-D, "(MANDATORY FOR CLUSTERING)"
    label_encoder = LabelEncoder()
    df['route'] = df[org_col].astype(str) + '-' + df[des_col].astype(str) 
    df['route_encoded'] = label_encoder.fit_transform(df['route'])
    return df

def standard_scaling(df,list_feat):
    #list_feat berupa list, contoh: ['A']
    scaler = StandardScaler()

    # Apply min-max scaling to the numeric columns
    df[list_feat] = scaler.fit_transform(df[list_feat])
    
    return df

def minmax_scaling(df,list_feat):
    #list_feat berupa list, contoh: ['A']
    scaler = MinMaxScaler(feature_range=(0, 10))

    # Apply min-max scaling to the numeric columns
    df[list_feat] = scaler.fit_transform(df[list_feat])
    
    return df

from scipy import stats
def remove_outliers(ts):
    print(f"lenght data before remove outliers : {len(ts)}")
    # Assuming you have a DataFrame named 'df' and you want to remove outliers from the 'column_name' column
    column_name = 'y'
    z_scores = np.abs(stats.zscore(ts[column_name]))
    threshold = 3  # Adjust the threshold as needed

    # Create a mask to identify outlier values
    outlier_mask = z_scores > threshold

    # Remove rows with outlier values
    ts= ts[~outlier_mask]
    ts.reset_index(inplace=True)
    print(f"lenght data after remove outliers : {len(ts)}")
    return ts

#cluster berdasarkan total penghasilan per rute
def add_clusters(df):
#     df = add_route(df,'kode_org','kode_des')
    feat =  ['y','route_encoded']
    #STEP 1: BIKIN DATAFRAME TENTANG SUM Y SETIAP RUTE TANPA TIMEFRAME
    
    route_df = df.groupby(["org","kode_org","des","kode_des","route_encoded"]).agg({'y': ["sum"]})
    # Rename the columns
    route_df.columns = ["y"]
    # Reset the index
    route_df = route_df.reset_index()
    
    
    #STEP 2(OPTIONAL):FEATURE SCALLING 
    route_df = standard_scaling(route_df,['y'])
#     print(f"ERROR CHECKER 1: {route_df}")
    
    #STEP 3 : FIND BEST K/N
    Elbow_M = KElbowVisualizer(KMeans(), k=10)
    Elbow_M.fit(route_df[feat]) 
    k = Elbow_M.elbow_value_
#     print(f"ERROR CHECKER 2: recomended k={k}")
    
    #STEP 4 : FIT CLUSTERING ALGORITHM
    AC = AgglomerativeClustering(n_clusters=k)
    yhat_AC = AC.fit_predict(route_df[feat])
    route_df['clusters']= yhat_AC
    
    #STEP 5 : MERGE TO MAIN DataFrame
    df = df.merge(route_df[['route_encoded','clusters']], on=['route_encoded'], how='left')
    return(df)
    
    
def create_holiday(df):
    ## dataframe hari libur
    id_holidays = holidays.ID(years=range(min(df.date.dt.year), max(df.date.dt.year)+1))

    data = []

    for day, event in id_holidays.items():
        data.append({'day': day, 'event': event})
        
    df_holidays = pd.DataFrame(data)
    df_holidays['day'] = pd.to_datetime(df_holidays['day'], format='%Y/%m/%d')

    #cuti idul fitri
    index = df_holidays.loc[df_holidays['event']=="Hari Raya Idul Fitri"].index
    original_date = df_holidays.loc[index, 'day']
    for date in original_date.values:
        date = pd.to_datetime(date)
    #     print(date)
        for i in range(1, 4):
    #         print(i)
            date_before = date - timedelta(days=i)
    #         print(date_before)
            new_row = {'event': 'Cuti Hari Raya Idul Fitri', 'day': date_before}
    #         print(new_row)
            df_holidays = pd.concat([df_holidays, pd.DataFrame([new_row])], ignore_index=True)

    index = df_holidays.loc[df_holidays['event']=="Hari kedua dari Hari Raya Idul Fitri"].index
    original_date = df_holidays.loc[index, 'day']
    for date in original_date.values:
        date = pd.to_datetime(date)
    #     print(date)
        for i in range(1, 3):
    #         print(i)
            date_after = date + timedelta(days=i)
    #         print(date_before)
            new_row = {'event': 'Cuti Hari Raya Idul Fitri', 'day': date_after}
    #         print(new_row)
            df_holidays = pd.concat([df_holidays, pd.DataFrame([new_row])])
    return df_holidays

def create_features(df,df_holidays):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['Is_Mon'] = (df['date'].dt.dayofweek == 0) *1
    df['Is_Tue'] = (df['date'].dt.dayofweek == 1) *1
    df['Is_Wed'] = (df['date'].dt.dayofweek == 2) *1
    df['Is_Thu'] = (df['date'].dt.dayofweek == 3) *1
    df['Is_Fri'] = (df['date'].dt.dayofweek == 4) *1
    df['Is_Sat'] = (df['date'].dt.dayofweek == 5) *1
    df['Is_Sun'] = (df['date'].dt.dayofweek == 6) *1
    df['Is_wknd'] = df['date'].dt.dayofweek // 4 # Fri-Sun are 4-6, Monday is 0 so this is valid
    
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in df_holidays['day'].values else 0)
    df['is_covid'] = ((df['date'] >= '2020-03-01') & (df['date'] <= '2021-12-31')).astype(int)
    
    df['Hijri_Year'] = df['date'].apply(lambda x: hijri_converter.Gregorian(x.year, x.month, x.day).to_hijri().year)
    df['Hijri_Month'] = df['date'].apply(lambda x: hijri_converter.Gregorian(x.year, x.month, x.day).to_hijri().month)
    df['Hijri_Day'] =df['date'].apply(lambda x: hijri_converter.Gregorian(x.year, x.month, x.day).to_hijri().day)
#     df['is_long'] = df.apply(determine_long_route, axis=1)
    
    return df

def add_lags(df):
    df.set_index('date', inplace=True)
    target_map = df['y'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('7 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('14 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('28 days')).map(target_map)
    df.reset_index(inplace=True)
    return df

def gs_LSTM(data,model):
    pass

def gs_RF(data,model):
    pass

def process_data(request):
    dataset = request.dataset
    outliers = request.outliers
    normalization = request.normalization
    method = request.method
    gridSearch = request.gridSearch
    period_start = request.period_start
    period_end = request.period_start
    
    if outliers != "No":
        outliers = request.outliers
    if normalization!= "No":
        normalization = request.normalization  
    if method != "No":
        method = request.method
    
    """
    main.py
    """
    df = create_df(dataset,'penghasilan_muatan','tanggal')
    df = add_route(df,'kode_org','kode_des')

    """
    Sum Target value by the same route+date
    """
    df = df.groupby(["date","org","kode_org","des","kode_des","route","route_encoded"]).agg({"y":  ["sum"]})
    df.columns = ["y"]
    df = df.reset_index()
    """
    End
    """

    df = add_clusters(df)
    df_holidays = create_holiday(df)
    df = create_features(df,df_holidays)
    df = add_lags(df)

    """
    Remove Outliers from timeseries
    """
    df = remove_outliers(df)
    
    """
    FEATURE SELECTIOn
    """
    features = (['kode_org',
                'kode_des',
                'clusters',
    #              'is_long',
                'is_holiday',
                'is_covid',
                'year',
                'month',
                'dayofweek','dayofmonth','dayofyear',
                'Is_Mon','Is_Tue','Is_Wed','Is_Thu','Is_Fri','Is_Sat','Is_Sun',
                'Hijri_Year','Hijri_Month','Hijri_Day',
                'lag1','lag2','lag3'])
    label = ['y']
    
    """
    Feature Scalling
    """
    df = standard_scaling(df,['lag1','lag2','lag3'])
    df.head()
    
    """
    MODEL
    """

    from xgboost import plot_importance, plot_tree
    X_train = df[features]
    y_train = df[label]
    model = xgb.XGBRegressor(n_estimators=500)
    model.fit(X_train, y_train)    
        
    return df, model, period_start,period_end, df_holidays