from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import time
start = time.time() # 컴파일 시작 시간


archive = pd.read_csv('./datasets/house_pricing.csv')
print(list(archive.head(0)))

feed = archive[['sqft_above', 'sqft_living', 'sqft_living15', 'grade', 'bathrooms', 'sqft_lot', 'sqft_lot15']] # Must be a 2D array
target  = archive['price']

# Nominate parameters
grid = {'n_estimators': [200, 500, 1000], 'learning_rate': [0.01, 0.1, 0.5], 'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5], 'max_depth': range(2, 5)}

# Set options for GridSearhCV
searcher = GridSearchCV(XGBRegressor(), grid, cv=5, n_jobs=4) # n_jobs: 병렬처리에 쓰이는 cpu core 개수

# Split the entire dataset into two groups: training set, test set
x_train, x_test, y_train, y_test = train_test_split(feed, target, random_state=0)

# Search the best parameters on training set
searcher.fit(x_train, y_train)

# Print out results
print("Best parameters: {}".format(searcher.best_params_))
print("Runtime: {}".format(time.time() - start)) # 컴파일에 걸린 시간