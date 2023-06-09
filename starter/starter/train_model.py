# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import pickle

# Add code to load in the data.
data = pd.read_csv('../data/census_clean.csv')
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
trained_model = train_model(X_train, y_train)

with open('../data/trained_model.pkl', 'wb') as f:
    pickle.dump(train_model, f)
with open('../data/encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
with open('../data/lb.pkl', 'wb') as f:
    pickle.dump(lb, f)

predicitons = inference(trained_model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, predicitons)
print('precision: ', precision)
print('recall: ', recall)
print('fbeta: ', fbeta)