# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

from dataset import load


data = load(2, ...)

X, y = data.data, data.target

print(X)
print(y)

# This dataset is way too high-dimensional. Better do PCA:
pca = PCA(n_components=2)

# Maybe some original features where good, too?
selection = SelectKBest(k=1)

# Build estimator from PCA and Univariate selection:

combined_features = FeatureUnion([('pca', pca), ('univ_select', selection)])

# Use combined features to transform dataset:
X_features = combined_features.fit(X, y).transform(X)

svm = SVC()

# Do grid search over k, n_components and C:

pipeline = Pipeline([('features', combined_features), ('svm', svm)])

param_grid = dict(
                features__pca__n_components=[1, 2, 3],
                features__univ_select__k=[1, 2],
                svm__C=[0.1, 1, 10],
            )

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
grid_search.fit(X, y)
print('==============================')
print(grid_search.best_estimator_)
