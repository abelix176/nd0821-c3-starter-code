from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

from sklearn.preprocessing import LabelBinarizer

from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI()

class InferenceRequest(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example" : {
                "age": 35,
                "workclass": "Self-emp-inc",
                "fnlgt": 250000,
                "education": "Doctorate",
                "education_num": 16,
                "marital_status": "Never-married",
                "occupation": "Prof-specialty",
                "relationship": "Not-in-family",
                "race": "Asian-Pac-Islander",
                "sex": "Female",
                "capital_gain": 5000,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "Japan"
            }
        }

# Categorical Features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@app.get("/")
def root():
    return {"message": "Welcome!"}

@app.post("/inference")
def perform_inference(request: InferenceRequest):
    """
    Perform model inference on the provided data.

    Parameters
    ----------
    request : InferenceRequest
        The request object containing the input data.

    Returns
    -------
    str
        The predicted label generated by the model.
    """
    df = pd.DataFrame([dict(request)])
    df.columns = df.columns.str.replace("_", "-")

    # Load model
    with open('starter/model/trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('starter/model/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('starter/model/lb.pkl', 'rb') as f:
        lb = pickle.load(f)

    # process data
    X_inf, _, _, _ = process_data(
        df, categorical_features=cat_features, training=False,
        encoder=encoder, lb=lb
    )

    # perform inference
    pred = inference(model, X_inf)
    pred = lb.inverse_transform(pred)[0]

    return pred
