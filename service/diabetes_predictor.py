"""
This file contains the machine learning model and data analysis. 
"""
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from typing import List
from sklearn.metrics import roc_curve, auc,recall_score,precision_score


def train_model() -> None:
    """Trains an XGBClassifier to predict diabetes.
    """
    data = get_samples()
    y = data['Outcome']
    x = data.drop(['Outcome'], axis=1)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=101
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

    y_val_pred = model.predict_proba(x_val)[:, 1]

    plot_roc(y_true=y_val, y_pred=y_val_pred)
    

    plot_features(x_train.columns, model.feature_importances_)

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
    print(f'Predicted probability {round(prediction*100)}\n %' if verbose else "", end="")
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


def plot_features(columns: list, importances: list, n=20) -> None:
    """Create plot of feature importances
    Args:
        columns (list): columns in data
        importances (list): importances from model
        n (int, optional): number of features. Defaults to 20.
    """
    df = (pd.DataFrame({"features": columns,
                        "feature_importance": importances})
          .sort_values("feature_importance", ascending=False)
          .reset_index(drop=True))

    sns.barplot(x="feature_importance",
                y="features",
                data=df[:n],
                orient="h")
    plt.tight_layout()
    plt.savefig("figures/features.png")
    plt.clf()


def plot_roc(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> None:
    """Plots ROC curve.

    Args:
        y_true (pd.DataFrame): True label
        y_pred (pd.DataFrame): Predicted probability
    """    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig('figures/roc.png')
    plt.clf()

if __name__ == "__main__":
    create_table()
    train_model()
