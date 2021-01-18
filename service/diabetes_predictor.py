import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.utils import resample


def train_model():
    data = pd.read_csv('data/diabetes.csv')

    """df_majority = data.loc[data.Outcome == 0].copy()
    df_minority = data.loc[data.Outcome == 1].copy()
    df_minority_upsampled = resample(
        df_minority, replace=True, n_samples=500, random_state=123) 
    data = pd.concat([df_majority, df_minority_upsampled])"""
    Y = data['Outcome']
    X = data.drop(['Outcome'], axis=1)
    x_train, x_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.1, random_state=101
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
    #results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    model.save_model('xgb.hdf5')
    print('XGB classifier saved')
    #print(f'Average cross-validation accuracy: {round(results.mean(), 4)}')


def predict_diabetes_probability(data: list[float]) -> float:
    xgb_model = XGBClassifier()
    xgb_model.load_model('xgb.hdf5')

    prediction = xgb_model.predict_proba(np.array(data).reshape(1, -1))
    print(prediction[0][1])
    return prediction


def create_data_report():
    from pandas_profiling import ProfileReport
    data = pd.read_csv('data/diabetes.csv')
    profile_title = "Generated Data Report"
    profile = ProfileReport(
        data, title=profile_title, explorative=True)
    profile.to_file("profile_report.html")
    return


if __name__ == "__main__":
    train_model()
    data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    predict_diabetes_probability(data)
