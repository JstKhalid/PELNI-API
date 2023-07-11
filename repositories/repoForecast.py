import schemas
import repositories.functions as functions
from fastapi import APIRouter, Depends, HTTPException, status
import xgboost as xgb
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np


def main(request: schemas.forecast):
    
    def forecast_cargo(request):
        dataset_cargo = "D://FrameworkDoRoute//PELNI//revenueDaily.xlsx"
        period_start = request.period_start
        period_end = request.period_start

        """
        main.py
        """
        df = functions.create_df(dataset_cargo,'penghasilan_muatan','tanggal')
        df = functions.add_route(df,'kode_org','kode_des')

        """
        Sum Target value by the same route+date
        """
        df = df.groupby(["date","org","kode_org","des","kode_des","route","route_encoded"]).agg({"y":  ["sum"]})
        df.columns = ["y"]
        df = df.reset_index()
        """
        End
        """

        df = functions.add_clusters(df)
        df_holidays = functions.create_holiday(df)
        df = functions.create_features(df,df_holidays)
        df = functions.add_lags(df)

        """
        Remove Outliers from timeseries
        """
        df = functions.remove_outliers(df)
        
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
        df = functions.standard_scaling(df,['lag1','lag2','lag3'])
        df.head()
        
        """
        MODEL
        """

        from xgboost import plot_importance, plot_tree
        X_train = df[features]
        y_train = df[label]
        model = xgb.XGBRegressor(n_estimators=500)
        model.fit(X_train, y_train)    
    
        # Assuming you have a DataFrame named df with columns 'route_encoded', 'kode_org', and 'kode_des'
        route_df = df.drop_duplicates(['route_encoded'])[['kode_org','kode_des','clusters']]
        # Create an empty list to store the new DataFrames
        dfs = []

        for org, des, cluster in route_df.values:
            date_range = pd.date_range(start=period_start, end=period_end, freq='D')
            # Create a new DataFrame with org, des, and date columns
            new_df = pd.DataFrame({'kode_org': org,
                                'kode_des': des,
                                'clusters':cluster,
                                'date': date_range})   
            # Append the new_df to the list of DataFrames
            dfs.append(new_df)
            
        # Concatenate all DataFrames in the list
        future = pd.concat(dfs, ignore_index=True)

        future['type_rev'] = 2
        future = functions.create_features(future,df_holidays)
        df['type_rev'] = 1
        ts_and_future = pd.concat([df, future])
        ts_and_future = functions.add_lags(ts_and_future)
        """
        X_Prediction
        """
        future_w_features = ts_and_future.loc[ts_and_future.type_rev==2].copy()
        future_w_features  = functions.standard_scaling(future_w_features ,['lag1','lag2','lag3'])
        """
        Prediction 
        """
        future_w_features['pred'] = model.predict(future_w_features[features])
        """
        OUTPUT
        """
        output_cargo = (future_w_features[['date','kode_org','kode_des','type_rev','pred']]).copy()
        # output_cargo[['departure','departure_time','jumlah_penumpang','voyage','total_pax']]='NULL'
        output_cargo = output_cargo.rename(columns={'date': 'departure_date','pred': 'revenue_cargo'})
        
        return output_cargo
    
    def forecast_pax(request):
        dataset_cargo = "D://FrameworkDoRoute//PELNI//RevenuePassengerDailyUpdated.xlsx"
        period_start = request.period_start
        period_end = request.period_start

        """
        main.py
        """
        df = functions.create_df(dataset_cargo,'penghasilan_penumpang','tanggal')
        df = functions.add_route(df,'kode_org','kode_des')

        """
        Sum Target value by the same route+date
        """
        df = df.groupby(["date","org","kode_org","des","kode_des","route","route_encoded"]).agg({"y":  ["sum"]})
        df.columns = ["y"]
        df = df.reset_index()
        """
        End
        """

        df = functions.add_clusters(df)
        df_holidays = functions.create_holiday(df)
        df = functions.create_features(df,df_holidays)
        df = functions.add_lags(df)

        """
        Remove Outliers from timeseries
        """
        df = functions.remove_outliers(df)
        
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
        df = functions.standard_scaling(df,['lag1','lag2','lag3'])
        df.head()
        
        """
        MODEL
        """

        from xgboost import plot_importance, plot_tree
        X_train = df[features]
        y_train = df[label]
        model = xgb.XGBRegressor(n_estimators=500)
        model.fit(X_train, y_train)    
    
        # Assuming you have a DataFrame named df with columns 'route_encoded', 'kode_org', and 'kode_des'
        route_df = df.drop_duplicates(['route_encoded'])[['kode_org','kode_des','clusters']]
        # Create an empty list to store the new DataFrames
        dfs = []

        for org, des, cluster in route_df.values:
            date_range = pd.date_range(start=period_start, end=period_end, freq='D')
            # Create a new DataFrame with org, des, and date columns
            new_df = pd.DataFrame({'kode_org': org,
                                'kode_des': des,
                                'clusters':cluster,
                                'date': date_range})   
            # Append the new_df to the list of DataFrames
            dfs.append(new_df)
            
        # Concatenate all DataFrames in the list
        future = pd.concat(dfs, ignore_index=True)

        future['type_rev'] = 2
        future = functions.create_features(future,df_holidays)
        df['type_rev'] = 1
        ts_and_future = pd.concat([df, future])
        ts_and_future = functions.add_lags(ts_and_future)
        """
        X_Prediction
        """
        future_w_features = ts_and_future.loc[ts_and_future.type_rev==2].copy()
        future_w_features  = functions.standard_scaling(future_w_features ,['lag1','lag2','lag3'])
        """
        Prediction 
        """
        future_w_features['pred'] = model.predict(future_w_features[features])
        """
        OUTPUT
        """
        output_pax = (future_w_features[['date','kode_org','kode_des','type_rev','pred']]).copy()
        # output_cargo[['departure','departure_time','jumlah_penumpang','voyage','total_pax']]='NULL'
        output_pax = output_pax.rename(columns={'date': 'departure_date','pred': 'revenue_pax'})
        
        return output_pax
    
    res_cargo = forecast_cargo(request)
    res_pax = forecast_pax(request)
    
    merged_df = res_pax.merge(res_cargo[['departure_date','kode_org','kode_des','revenue_cargo','type_rev']], on=['departure_date','kode_org','kode_des','type_rev'], how='outer')
    merged_df[['revenue_cargo','revenue_pax']] = merged_df[['revenue_cargo','revenue_pax']].fillna(0)
    
    res = []
    for _, row in merged_df.iterrows():
        pred_date = row['departure_date']
        column1 = row['kode_org']
        column2 = row['kode_des']
        column3 = row['type_rev']
        column4 = row['revenue_cargo']
        column5 = row['revenue_pax']
        
        this_res = {
            'departure_date': pred_date,
            'kode_org': column1,
            'kode_des': column2,
            'type_rev': column3,
            'revenue_cargo': column4,
            'revenue_pax': column5
        }
        
        res.append(this_res)
    
    return {
        "status": {
            "responseCode": status.HTTP_200_OK,
            "responseDesc": "Success",
            "responseMessage": "Success fetching data!"
        },
        "result": res
    }