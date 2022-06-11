from sklearn.linear_model import LogisticRegression
from data_preparation import load
from src.training import select_model

if __name__ == '__main__':
    digit1, digit2 = 8, 5
    dataset = load(digit1, digit2)
    parameters = {'solver': ['saga', 'lbfgs', 'liblinear', 'newton-cg', 'sag'], 'max_iter': [100]}
    select_model(dataset, LogisticRegression, parameters)
