

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
...

## 🏁 Getting Started <a name = "getting_started"></a>

All requirements are listed in the 'requirements.txt'-file, simply run the following commands:

```
sudo apt-get install python3.8.2
sudo apt-get install python3-pip
git clone https://github.com/Sandbergo/online-diabetes-predictor.git
cd online-diabetes-predictor
```
now, to activate the frontend locally:
```
npm install -g serve
npm run build
serve -s build -l 3000
```
Then, activate the backend:
```
virtualenv env
source env/bin/activate 
pip install -r requirements.txt
FLASK_APP=app.py flask run
```

## ⛏️ Built Using <a name = "built_using"></a>
- [Python](https://www.python.org/) 
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
    
    
## ✍️ Authors <a name = "authors"></a>
- Lars Sandberg [@Sandbergo](https://github.com/Sandbergo)

## 🎉 Acknowledgements
https://towardsdatascience.com/create-a-complete-machine-learning-web-application-using-react-and-flask-859340bddb33
