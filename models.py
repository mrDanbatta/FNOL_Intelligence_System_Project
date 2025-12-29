import pandas as pd
import numpy as np
import joblib
import os
from huggingface_hub import hf_hub_download
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


MODEL_PATH = "models/best_random_forest_model.pkl"
FEATURES_PATH = "models/feature_columns.pkl"

FEATURES = ['Claim_Type', 
            'Estimated_Claim_Amount',
            'Traffic_Condition',
            'Weather_Condition',
            'Driver_Age',
            'License_Years',
            'Vehicle_Type',
            'Vehicle_Year']

REPO_ID = "MrDanbatta/FNOL_Intelligence_System"
MODEL_FILE_NAME = "best_random_forest_model.pkl"
FEATURES_FILE_NAME = "feature_columns.pkl"

# def load_model():
#     """Load the pre-trained Random Forest model from disk."""
#     with open(MODEL_PATH, 'rb') as file:
#         model = joblib.load(file)
#     with open(FEATURES_PATH, 'rb') as file:
#         feature_columns = joblib.load(file)
#     return model, feature_columns

def load_model():
    """Load model and feature columns from Hugging Face Hub."""
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILE_NAME
    )
    features_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FEATURES_FILE_NAME
    )

    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)
    
    return model, feature_columns

def save_model(model, feature_columns):
    """Save the trained Random Forest model to disk."""
    with open(MODEL_PATH, 'wb') as file:
        joblib.dump(model, file)
    with open(FEATURES_PATH, 'wb') as file:
        joblib.dump(feature_columns, file)




def data_pipeline(df):
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # define numeric pipeline
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # define categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ]).set_output(transform="pandas")


    # apply preprocessing to the claims dataframe

    df_processed = preprocessor.fit_transform(df)
    return df_processed


def winsorize(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df


def retrain_model(new_data):

    data = new_data.copy()

    #derived features
    data["Driver_Age"] = (pd.to_datetime(data["Accident_Date"]) - pd.to_datetime(data["Date_of_Birth"])).dt.days // 365
    data["License_Years"] = (pd.to_datetime(data["Accident_Date"]) - pd.to_datetime(data["Full_License_Issue_Date"])).dt.days // 365
    data["FNOL_Delay"] = (pd.to_datetime(data["FNOL_Date"]) - pd.to_datetime(data["Accident_Date"])).dt.days
    data["Settlement_Days"] = (pd.to_datetime(data["Settlement_Date"]) - pd.to_datetime(data["Accident_Date"])).dt.days

    #fix outliers
    outlier_columns = ['Estimated_Claim_Amount',
                       'Ultimate_Claim_Amount',
                       'FNOL_Delay',
                       'Settlement_Days',]
    
    for col in outlier_columns:
        data = winsorize(data, col)

    data['Ultimate_Claim_Amount'] = np.log1p(data['Ultimate_Claim_Amount'])

    #features and target
    featues = [
        'Claim_Type', 
        'Estimated_Claim_Amount',
        'Traffic_Condition',
        'Weather_Condition',
        'Driver_Age',
        'License_Years',
        'Vehicle_Type',
        'Vehicle_Year'
    ]
    target = 'Ultimate_Claim_Amount'

    data = data[featues + [target]]

    # load production model
    prod_model = load_model()
    prod_rf_model = prod_model[0]

    expected_features = list(prod_model[1])  # feature columns used in the production model
    data_processed = data_pipeline(data)
    data_processed = data_processed.reindex(columns=expected_features, fill_value=0)
    X = data_processed
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    new_model = RandomForestRegressor(**prod_rf_model.get_params())
    new_model.fit(X_train, y_train)

    y_pred = prod_rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    y_pred_new = new_model.predict(X_test)
    rmse_new = np.sqrt(mean_squared_error(y_test, y_pred_new))

    #promote if better
    promoted = False
    if rmse_new < rmse:
        save_model(new_model, expected_features)
        promoted = True

    return {
        "old_model_rmse": rmse,
        "new_model_rmse": rmse_new,
        "promoted": promoted
    }
   


