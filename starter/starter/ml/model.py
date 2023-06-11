from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from starter.starter.ml.data import process_data
import logging


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def evaluate_model_slices(model, df, categorical_features, encoder, lb, log_path):
    """
    Perform model evaluation on data slices of the categorical features.
    
    Inputs
    ------
    model: trainded model to be evaluated
    df: pandas datafram of data to be used
    categorical_features: categorical features of the input data
    encoder: encoder used to train the model
    lb: binarizer
    log_path: path to wirte output to

    Outputs
    ------
    None
    """
    logging.basicConfig(filename = log_path,
                    level=logging.INFO, format="%(asctime)-15s %(message)s")
    logger = logging.getLogger()

    for feature in categorical_features:
        for value in df[feature].unique():
            df_sliced = df[df[feature] == value]

            X_test, y_test, _, _ = process_data(
                df_sliced, categorical_features=categorical_features, label="salary", training=False,
                encoder=encoder, lb=lb
            )
            predicitons = inference(model, X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, predicitons)

            logger.info(f"Slice: {feature}={value}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"fbeta: {fbeta}")
            logger.info(f"-----------------------")


