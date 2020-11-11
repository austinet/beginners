from aiclasslib import datafit, graph
from danakit import *
import warnings
warnings.filterwarnings('ignore')

datafit = datafit()
graph = graph()
regressor = regressor()
classifier = classifier()
ensemble = ensemble()

archive = datafit.read("./datasets/house_pricing.csv")
datafit.show(archive, 0)

datafit.drop(archive, ["id", "date"])
datafit.show(archive, 0)
print()

# regressor.linear(archive, ['sqft_living', 'bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'waterfront', 'view',
#              'grade','yr_built','zipcode'], ["price"])
# print()
# regressor.ridge(archive, ['sqft_living', 'bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'waterfront', 'view',
#              'grade','yr_built','zipcode'], ["price"])
# print()
# regressor.lasso(archive, ['sqft_living', 'bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'waterfront', 'view',
#              'grade','yr_built','zipcode'], ["price"])
# print()
# regressor.elastic_net(archive, ['sqft_living', 'bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'waterfront', 'view',
#              'grade','yr_built','zipcode'], ["price"])
print()
ensemble.random_forest_regressor(archive, ['sqft_living', 'bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'waterfront', 'view',
               'grade','yr_built','zipcode'], ["price"])
print()
ensemble.adaboost_regressor(archive, ['sqft_living', 'bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'waterfront', 'view',
               'grade','yr_built','zipcode'], ["price"])

# classifier.logistic_regression(archive, ["SepalLengthCm", "SepalWidthCm"], ["Species"])
# print()
# classifier.support_vector_machines(archive, ["SepalLengthCm", "SepalWidthCm"], ["Species"])
# print()
# classifier.k_nearest_neighbors(archive, ["SepalLengthCm", "SepalWidthCm"], ["Species"])
# print()
# classifier.decision_trees(archive, ["PetalLengthCm", "PetalWidthCm"], ["Species"])
# print()
# ensemble.random_forest_classifier(archive, ["PetalLengthCm", "PetalWidthCm"], ["Species"])
# print()
# ensemble.adaboost_classifier(archive, ["PetalLengthCm", "PetalWidthCm"], ["Species"])