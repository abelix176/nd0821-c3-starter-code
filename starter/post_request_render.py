import requests

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

url = "https://fastapi-luh9.onrender.com/inference"

response = requests.post(url=url, json=data)
print(response.status_code)
print(response.json())