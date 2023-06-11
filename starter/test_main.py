import json
from fastapi.testclient import TestClient
from starter.main import app

client = TestClient(app)

def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome!"}

def test_post_inference_prediction_1():
    data = {
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
    response = client.post("/inference", json=data)
    assert response.status_code == 200
    assert response.text.strip() == '"<=50K"'

def test_post_inference_prediction_2():
    data = {
        "age": 45,
        "workclass": "Private",
        "fnlgt": 260000,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 8000,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    }
    response = client.post("/inference", json=data)
    assert response.status_code == 200
    assert response.text.strip() == '">50K"'
