

<h1 align="center">Online Diabetes Predictor</h1>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]() 
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center">
:hospital: Predicting the possibility of diabetes from physiological data 
</p>
<br> 

## 🧐 About <a name = "about"></a>
Simple Web app for requesting the evaluation of a Machine Learning model of the probability of the patient suffering from diabetes. 

## 🏁 Getting Started <a name = "getting_started"></a>

To run the app locally, follow these instructions:

```
git clone https://github.com/Sandbergo/online-diabetes-predictor.git
cd online-diabetes-predictor
```
now, to activate the frontend locally:
```
cd frontend
npm install -g serve
npm run build
serve -s build -l 3000
```
Then, activate the api:
```
cd backend
sudo apt-get install python3.8.2
sudo apt-get install python3-pip
pip install virtualenv
virtualenv env
source env/bin/activate 
pip install -r requirements.txt
python diabetes_predictor.py
python app.py
```

## ⛏️ Built Using <a name = "built_using"></a>
- [Python](https://www.python.org/) 
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/)
    
## ✍️ Authors <a name = "authors"></a>
- Lars Sandberg [@Sandbergo](https://github.com/Sandbergo)

## 🎉 Acknowledgements
- Kharan Baharot Flask + React template [@kb22/ML-React-App-Template](https://github.com/kb22/ML-React-App-Template)
