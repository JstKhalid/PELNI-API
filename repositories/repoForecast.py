import schemas
import repositories.functions as functions
from fastapi import APIRouter, Depends, HTTPException, status, Response
import xgboost as xgb

def main(request:schemas.forecast):
    df, X_pred, pred_date = functions.process_data(request)
    
    distinct_route = df[['kode_org', 'org', 'kode_des', 'des']].drop_duplicates()
    res = []

    for row in distinct_route.values:
        # Extract data for the current route
        data = df.loc[(df['kode_org'] == row[0]) & (df['kode_des'] == row[2])]
        
        # Apply optional functions based on request parameters
        if request.outliers == "yes":
            data = functions.ElimOutliers(data)
        else:
            pass
        
        if request.normalization == "yes":
            data = functions.normalisasi(data)
        else:
            pass
        
        # Choose the appropriate method and perform grid search if requested
        if request.method == "XGB":
            model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            if request.gridSearch == "yes":
                model = functions.gs_XGB(data, model)
        elif request.method == "LSTM":
            model = functions.gs_LSTM(data, model)
        else:
            model = functions.gs_RF(data, model)
        
        # Prepare X and y for training
        # X_train, X_test, y_train, y_test = functions.split_data(data)
        
        #-----------------------------------------------------------
        #UNTUK SEMENTARA PAKAI DATA DUMMY IF NORMALISASI = NO
        X = data[['year','month','weekofmonth','weekday','day','hijri_year','hijri_month','hijri_day','is_holiday','is_covid']]
        y = data['total_revenue']
        
        # Fit the model
        # model.fit(X_train, y_train)
        model.fit(X,y)
        
        
        # Predict the target value for X_pred
        y_pred = model.predict(X_pred)
        y_pred = float(y_pred)  # Convert numpy.float32 to Python float
        
        # Store the prediction result for the current route
        this_res = [pred_date, row[1], row[0], row[3], row[2], y_pred]
        res.append(this_res)

        dict_res = [{'date': a[0], 'org': a[1], 'kode_org': a[2], 'des': a[3], 'kode_des': a[4], 'y_pred': a[5]} for a in res]

    return {
        "status": {
            "responseCode": status.HTTP_200_OK,
            "responseDesc": "Success",
            "responseMessage": "Success fetching data!"
        },
        "result": dict_res
    }