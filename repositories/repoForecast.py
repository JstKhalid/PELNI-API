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

def main(request:schemas.forecast):
    #Main Parameter
    dataset = request.dataset #nanti di ganti get_json/get_request['dataset'] klo udah tau cara upload dataset by request
    #set optional parameter sesuai request (kalau ada)
    tanggal= request.tanggal
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
        
    #read dataset 
    df = pd.read_excel(dataset)
    #format column tidak di perlukan kalau ada fungsi yang bisa memformat langsung data dari retrieve api pelni
    df = df.rename(columns={'tanggal': 'date'})
    df = df.rename(columns={'penghasilan_penumpang': 'y'})
    #selalu di butuhkan
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    
    #table hari libur (sementara sebelum stretch nataru sama idul fitri)
    id_holidays = holidays.ID(years=range(min(df.date.dt.year), max(df.date.dt.year)+1))

    data = []

    for day, event in id_holidays.items():
        data.append({'day': day, 'event': event})
        
    df_holidays = pd.DataFrame(data)
    df_holidays['day'] = pd.to_datetime(df_holidays['day'], format='%Y/%m/%d')
    
    #convert pred_date menjadi feature
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
    
    #aggregasi y(revenue/pax) jumlahnya lebih dari satu di (satu tanggal dan rute) yang sama **(optional)
    df = df.groupby(["date","des","kode_des","org","kode_org"]).agg({"y": ["sum", "mean","count"]})
    # Rename the columns
    df.columns = ["total_revenue", "avg_revenue","ship_count"]
    # Reset the index
    df = df.reset_index()
    
    #stretch date menjadi fitur-fitur baru
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['weekofmonth'] = df['date'].apply(lambda x: (x.day-1) // 7 + 1) # calculate week of month
    df['hijri_year'] = df['date'].apply(lambda x: hijri_converter.Gregorian(x.year, x.month, x.day).to_hijri().year)
    df['hijri_month'] = df['date'].apply(lambda x: hijri_converter.Gregorian(x.year, x.month, x.day).to_hijri().month)
    df['hijri_day'] = df['date'].apply(lambda x: hijri_converter.Gregorian(x.year, x.month, x.day).to_hijri().day)
    df['is_holiday']=df['date'].apply(lambda x: 1 if x in df_holidays['day'].values else 0)
    df['is_covid'] = ((df['date'] >= '2020-03-01') & (df['date'] <= '2021-12-31')).astype(int)
    
    #set X_pred isi X_pred bisa berubah tergantung hasil akhir riset ml nanti
    X_pred = np.array([[year, month, weekofmonth, weekday, day, hijri_year, hijri_month, hijri_day, is_holiday, is_covid]])
    # masi belom flexible dan sangat bergantung pada proses agregasi dan rename setelah read dataset
    
    
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
        steps = int(0.1*len(data))

        X_train = X[:-steps]
        X_test  = X[-steps:]

        y_train = y[:-steps]
        y_test  = y[-steps:]
        return X_train,X_test,y_train,y_test
    
    def gs_XGB(X_train,y_train,model):
        params = {
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 500, 1000]
        }

        # Perform GridSearchCV to find the best hyperparameters
        grid_search = GridSearchCV(model, param_grid=params, cv=5)
        grid_search.fit(X_train, y_train)
        
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **grid_search.best_params_)
        return model
    
    def gs_LSTM(data,model):
        pass
    
    def gs_RF(data,model):
        pass
    
    
    res = [] #nyimpan hasil yang nantinya dipake buat jadi output ke database/response
    distinct_route = df[['kode_org','org','kode_des','des']].drop_duplicates() #cek semua rute yang ada

    for row in distinct_route.values: #iterasi berdasarkan jumlah rute yg tersedia
        data = df.loc[(df['kode_org'] == row[0])&(df['kode_des'] == row[2])]
        #check apakah ada parameter optional
        if outliers == "yes":
            #panggil outliers(data)
            ElimOutliers(data)
        else:
            pass
        
        if normalization == "yes":
            #panggil normalization(data)
            normalisasi(data)
        else:
            pass

        
        #feature extraction (masi sementara ada kemungkinan pake kondisi beradasarkan corr table/grafik berdasarkan persentage atau warmnest)
        X = data[['year','month','weekofmonth','weekday','day','hijri_year','hijri_month','hijri_day','is_holiday','is_covid']]
        y = data['total_revenue']
        
        if method == "XGB":
            model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            if gridSearch =="yes":
                #panggil split_data(data)
                # split_data(data)
                # X_train,X_test,y_train,y_test = split_data(data)
                #panggil gs_method
                # gs_XGB(X_train,y_train,model)
                pass
        elif method == "LSTM":
            pass
        else:
            pass
        
        model.fit(X,y)
        y_pred = model.predict(X_pred)
        y_pred = float(y_pred)  # Convert numpy.float32 to Python float
        this_res = [pred_date, row[1], row[0], row[3], row[2], y_pred] #menyimpan hasil prediksi per rute
        print(this_res)
        res.append(this_res)
    
    dict_res = [{'date':a[0], 'org': a[1], 'kode_org': a[2], 'des': a[3], 'kode_des':a[4], 'y_pred':a[5]} for a in res]
    
    # jsons = json.dumps(dict_res, default=str)
    
    return {
        "status": {
            "responseCode": status.HTTP_200_OK,
            "responseDesc": "Success",
            "responseMessage": "Success fetching data!"
        },
        "result": dict_res
    }