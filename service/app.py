from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
from diabetes_predictor import predict_diabetes_probability


flask_app = Flask(__name__)
app = Api(app=flask_app, version="1.0",
          title="Online Diabetes Predictor",
          description="Online Diabetes Predictor")

name_space = app.namespace('prediction', description='Prediction APIs')

# Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
model = app.model('Prediction params',
    {'Pregnancies': fields.Float(
        required=True, description="Pregnancies", help="Number of pregnancies"),
     'Glucose': fields.Float(
        required=True, description="Glucose", help="Glucose level"),
     'BloodPressure': fields.Float(
        required=True, description="BloodPressure", help="Blood pressure"),
     'SkinThickness': fields.Float(
        required=True, description="SkinThickness", help="Skin thickness"),
     'Insulin': fields.Float(
        required=True, description="Insulin", help="Insulin level"),
     'BMI': fields.Float(
        required=True, description="BMI", help="Body Mass Inex"),
     'DiabetesPedigreeFunction': fields.Float(
        required=True, description="DiabetesPedigreeFunction", help="Diabetes pedigree function"),
     'Age': fields.Float(
        required=True, description="Age", help="Age")})


@name_space.route("/")
class MainClass(Resource):

    def options(self):
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

    @app.expect(model)
    def post(self):
        print('response requested')
        try:
            formData = request.json
            data = [float(val) for val in formData.values()]
            print('data: ', data)
            prediction = predict_diabetes_probability(data)

            response = jsonify({
                "statusCode": 200,
                "status": "Prediction made",
                "result": f"Probability of diabetes: {round(prediction*100)} %"
                })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        except Exception as error:
            return jsonify({
                "statusCode": 500,
                "status": "Could not make prediction",
                "error": str(error)
            })
