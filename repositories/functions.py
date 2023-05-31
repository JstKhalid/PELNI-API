import numpy as np
import pandas as pd
import holidays
from datetime import datetime, date, timedelta
import hijri_converter
from hijri_converter import convert
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from fastapi import APIRouter, Depends, HTTPException, status, Response
import schemas
from sklearn.preprocessing import StandardScaler
import json


def read_dataset(dataset):
    df = pd.read_excel(dataset)
    df = df.rename(columns={'tanggal': 'date'})
    df = df.rename(columns={'penghasilan_penumpang': 'y'})
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    return df

def create_holidays_table(df):
    id_holidays = holidays.ID(years=range(min(df.date.dt.year), max(df.date.dt.year)+1))
    data = [{'day': day, 'event': event} for day, event in id_holidays.items()]
    df_holidays = pd.DataFrame(data)
    df_holidays['day'] = pd.to_datetime(df_holidays['day'], format='%Y/%m/%d')
    return df_holidays

def get_features(pred_date, df_holidays):
    year = pred_date.year
    month = pred_date.month
    day = pred_date.day
    weekday = pred_date.weekday()
    weekofmonth = (day - 1) // 7 + 1
    hijri_date = hijri_converter.Gregorian(year, month, day).to_hijri()
    hijri_year = hijri_date.year
    hijri_month = hijri_date.month
    hijri_day = hijri_date.day
    is_holiday = 1 if pred_date in df_holidays['day'].values else 0
    is_covid = 1 if (pred_date >= date(2020, 3, 1)) and (pred_date <= date(2021, 12, 31)) else 0
    return year, month, day, weekday, weekofmonth, hijri_year, hijri_month, hijri_day, is_holiday, is_covid

def aggregate_revenue(df):
    df = df.groupby(["date", "des", "kode_des", "org", "kode_org"]).agg({"y": ["sum", "mean", "count"]})
    df.columns = ["total_revenue", "avg_revenue", "ship_count"]
    df = df.reset_index()
    return df

def stretch_date_features(df, df_holidays):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['weekofmonth'] = df['date'].apply(lambda x: (x.day-1) // 7 + 1)
    df['hijri_year'] = df['date'].apply(lambda x: hijri_converter.Gregorian(x.year, x.month, x.day).to_hijri().year)
    df['hijri_month'] = df['date'].apply(lambda x: hijri_converter.Gregorian(x.year, x.month, x.day).to_hijri().month)
    df['hijri_day'] = df['date'].apply(lambda x: hijri_converter.Gregorian(x.year, x.month, x.day).to_hijri().day)
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in df_holidays['day'].values else 0)
    df['is_covid'] = ((df['date'] >= '2020-03-01') & (df['date'] <= '2021-12-31')).astype(int)
    return df

def prepare_X_pred(year, month, weekofmonth, weekday, day, hijri_year, hijri_month, hijri_day, is_holiday, is_covid):
    X_pred = np.array([[year, month, weekofmonth, weekday, day, hijri_year, hijri_month, hijri_day, is_holiday, is_covid]])
    return X_pred

def process_data(request):
    dataset = request.dataset
    tanggal = request.tanggal
    date_format = "%d-%m-%Y"
    tanggal = datetime.strptime(tanggal, date_format)
    pred_date = tanggal.date()
    outliers = request.outliers
    normalization = request.normalization
    method = request.method
    gridSearch = request.gridSearch
    
    if outliers != "No":
        outliers = request.outliers
    if normalization!= "No":
        normalization = request.normalization  
    if method != "No":
        method = request.method
    
    df = read_dataset(dataset)
    df_holidays = create_holidays_table(df)
    year, month, day, weekday, weekofmonth, hijri_year, hijri_month, hijri_day, is_holiday, is_covid = get_features(pred_date, df_holidays)
    df = aggregate_revenue(df)
    df = stretch_date_features(df,df_holidays)
    X_pred = prepare_X_pred(year, month, weekofmonth, weekday, day, hijri_year, hijri_month, hijri_day, is_holiday, is_covid)
    
    return df, X_pred, pred_date

def ElimOutliers(data):
    n = 0
    x = 1
    while x != 0:   
        n+=1
        q1 = data['total_revenue'].quantile(0.25)
        q3 = data['total_revenue'].quantile(0.75)
        iqr = q3 - q1

        # identify outliers
        lower_fence = q1 - 1.5*iqr
        upper_fence = q3 + 1.5*iqr
        outliers = data.loc[(data['total_revenue'] < lower_fence) | (data['total_revenue'] > upper_fence)]
        x= len(outliers)
#         if x > 0:
#             print("Outliers ke-",n,": ",outliers)
        # remove outliers
        data = data.loc[(data['total_revenue'] >= lower_fence) & (data['total_revenue'] <= upper_fence)]
    
    return data
    
# masi belom flexible dan sangat bergantung pada proses agregasi dan rename setelah read dataset
def normalisasi(data):
    unscaled = data['total_revenue'].values.reshape(len(data['total_revenue']),1)
    scaler = StandardScaler()
    scaler.fit(unscaled)
    scaled = scaler.transform(unscaled)
    data['total_revenue'] = scaled
    return  data    

def split_data(data):
    steps = int(0.1 * len(data))

    # feature extraction (masi sementara ada kemungkinan pake kondisi beradasarkan corr table/grafik berdasarkan persentage atau warmnest)
    X = data[['year', 'month', 'weekofmonth', 'weekday', 'day', 'hijri_year', 'hijri_month', 'hijri_day', 'is_holiday', 'is_covid']]
    y = data['total_revenue']

    print("Length of data:", len(data))
    print("Filtered X DataFrame:", X)
    print("Filtered y DataFrame:", y)

    X_train = X[:-steps]
    X_test = X[-steps:]

    y_train = y[:-steps]
    y_test = y[-steps:]

    print("Length of X_train:", len(X_train))
    print("Length of y_train:", len(y_train))

    return X_train, X_test, y_train, y_test

def gs_XGB(X_train, y_train):
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    params = {
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 500, 1000]
    }
    
    grid_search = GridSearchCV(model, param_grid=params, cv=5)
    grid_search.fit(X_train, y_train)
    
    best_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **grid_search.best_params_)
    
    return best_model

def gs_LSTM(data,model):
    pass

def gs_RF(data,model):
    pass
