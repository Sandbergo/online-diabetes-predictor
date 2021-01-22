"""
FastAPI based API for predicting diabetes, relies on diabetes_predictor.py.
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diabetes_predictor import predict_diabetes_probability


app = FastAPI(
    title="Online Diabetes Predictor",
    description="Predicts probability of diabetes from physiological data",
    version="0.9.0",
)

origins = [
    "http://localhost:3000",
    "localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class params(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


class response(BaseModel):
    statusCode: int
    status: str
    result: str


@app.post("/prediction/", response_model=response, status_code=200)
def get_prediction(payload: params):
    """
    Returns JSON response upon prediction request
    """
    print('response requested')
    try:
        formData = payload.dict()
        data = [float(val) for val in formData.values()]
        prediction = predict_diabetes_probability(data)

        response = {
            "statusCode": 200,
            "status": "Prediction made",
            "result": f"Probability of diabetes: {round(prediction*100)} %"
            }
        return response
    except Exception as error:
        # print(f'Error: \n{error}')
        return {
            "statusCode": 500,
            "status": "Could not make prediction",
            "error": str(error)
        }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
