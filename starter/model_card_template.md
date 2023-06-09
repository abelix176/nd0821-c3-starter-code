# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a Random Forest Classifier from sklear with the default hyperparameters.
## Intended Use
This model can be used to infer a salary given a number of input features, such as age, occupation, etc.
## Training Data
The training data is a 80% split from the overall Census data. (https://archive.ics.uci.edu/ml/datasets/census+income)
## Evaluation Data
The test data is a 20% split from the overall Census data.
## Metrics
The model performance was evaluated using precision, recall and fbeta:
- precision:  0.72
- recall:  0.63
- fbeta:  0.67

## Ethical Considerations
The dataset contains information about race and gender which may, under some circumstances, lead to an unfair scoring and could be considered unethical.
## Caveats and Recommendations
While quite extensive, the dataset dates back to 1994 and any predictions and implications for todays salaries may be imprecise.