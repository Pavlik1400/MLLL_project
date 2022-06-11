from sklearn.linear_model import LogisticRegression
from data_preparation import load

if __name__ == '__main__':
    data = load(8, 5)
    regressor = LogisticRegression(solver='saga', max_iter=200)
    regressor.fit(data.train_data, data.train_targets)

    train_score = regressor.score(data.train_data, data.train_targets)
    cv_score = regressor.score(data.cv_data, data.cv_targets)
    test_score = regressor.score(data.test_data, data.test_targets)

    print(f"Train score: {train_score}")
    print(f"Cross-validation score: {cv_score}")
    print(f"Test score: {test_score}")
