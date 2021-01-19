

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
cd ui
npm install -g serve
npm run build
serve -s build -l 3000
```
Then, activate the backend:
```
cd service
sudo apt-get install python3.8.2
sudo apt-get install python3-pip
pip install virtualenv
virtualenv env
source env/bin/activate 
pip install -r requirements.txt
python diabetes_predictor.py
FLASK_APP=app.py flask run
```

## ⛏️ Built Using <a name = "built_using"></a>
- [Python](https://www.python.org/) 
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
- [Flask](https://flask.palletsprojects.com/en/1.1.x/)
- [React](https://reactjs.org/)
    
## ✍️ Authors <a name = "authors"></a>
- Lars Sandberg [@Sandbergo](https://github.com/Sandbergo)

## 🎉 Acknowledgements
https://towardsdatascience.com/create-a-complete-machine-learning-web-application-using-react-and-flask-859340bddb33
