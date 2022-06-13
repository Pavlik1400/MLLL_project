from sklearn.svm import SVC

from data_preparation import load
from training import select_model

if __name__ == '__main__':
    digit1, digit2 = 8, 5
    dataset = load(digit1, digit2)
    parameters = {
        'kernel': ['sigmoid', 'rbf', 'poly'],
        'degree': [3, 4, 5],
    }
    select_model(dataset, SVC, parameters)
