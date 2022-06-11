from sklearn.neighbors import KNeighborsClassifier

from data_preparation import load
from training import select_model

if __name__ == '__main__':
    digit1, digit2 = 8, 5
    dataset = load(digit1, digit2)
    parameters = {'n_neighbors': list(range(1, 15))}
    select_model(dataset, KNeighborsClassifier, parameters)




