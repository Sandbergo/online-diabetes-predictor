"""
This file contains the machine learning model and data analysis. 
"""
import os
import sqlite3
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from typing import List


def train_model() -> None:
    """Trains an XGBClassifier to predict diabetes.
    """
    data = get_samples()
    y = data['Outcome']
    x = data.drop(['Outcome'], axis=1)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.1, random_state=101
        )
    model = XGBClassifier(
        n_estimators=100, max_depth=11, learning_rate=0.1,
        use_label_encoder=False, verbosity=0, 
        scale_pos_weight=1.8)
    model.fit(
        x_train, y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        eval_metric='auc',
        early_stopping_rounds=8,
        verbose=True,
        )
    
    model.save_model('xgb.hdf5')
    print('XGB classifier saved')


def predict_diabetes_probability(data: List[float], verbose=True) -> float:
    """Returns the model's predicted probability of diabetes from the given 
    physiological data.

    Args:
        data (List[float]): A list of the physiological data.

    Returns:
        float: Predicted probability of diabetes,
    """
    print('Predicting...\n' if verbose else "", end="")
    xgb_model = XGBClassifier()
    print('Loading model...\n' if verbose else "", end="")
    xgb_model.load_model('xgb.hdf5')
    print('Making Prediction...\n' if verbose else "", end="")
    prediction = xgb_model.predict_proba(np.array(data).reshape(1, -1))[0][1]
    print(f'Predicted probability {round(prediction*100)}\n' if verbose else "", end="")
    return prediction


def create_data_report() -> None:
    """
    Creates a pandas profiling report of the data.
    """
    from pandas_profiling import ProfileReport
    data = pd.read_csv('data/diabetes.csv')
    profile_title = "Generated Data Report"
    profile = ProfileReport(
        data, title=profile_title, explorative=True)
    profile.to_file("profile_report.html")
    return


def create_table() -> None:
    """Creates a database from csv.
    """
    open('data/diabetes.db', 'a').close()
    conn = sqlite3.connect('data/diabetes.db')
    data = pd.read_csv('data/diabetes.csv')
    data.to_sql('samples', conn, if_exists='replace', index=False)

    return


def get_samples() -> pd.DataFrame:
    """Get samples from database.

    Returns:
        pd.DataFrame: samples from the dataset.
    """
    conn = sqlite3.connect('data/diabetes.db')
    data = pd.read_sql('''SELECT * FROM samples''', conn)

    return data


if __name__ == "__main__":
    create_table()
    train_model()
