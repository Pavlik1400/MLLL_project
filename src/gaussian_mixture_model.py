import numpy as np
from sklearn.mixture import GaussianMixture
from data_preparation import load
from sklearn.decomposition import PCA

if __name__ == '__main__':
    data = load(8, 5)

    reducer = PCA(n_components=0.99)
    train_data = reducer.fit_transform(data.train_data)
    print(train_data.shape)
    model = GaussianMixture(n_components=2, covariance_type='full', max_iter=100)

    model.fit(train_data)

    cv_pred = model.predict(reducer.transform(data.cv_data))

    print(np.mean(data.cv_targets == cv_pred))

