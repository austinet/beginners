"""
RegressorKit v1.0

라이브러리 설명
---
    인공지능 과제를 위해 작성된 회귀 모델 라이브러리입니다.
    네 개의 선형 모델, 세 개의 비선형 모델, 네 개의 앙상블 모델이 포함되어 있습니다.

선형 모델
---
    1. Linear Regression
    2. Ridge Regression
    3. Lasso Regression
    4. Elastic-net Regression

비선형 모델
---
    1. K-Nearest Neighbors Regression
    2. Support Vector Machine Regression
    3. Decision Tree Regression

앙상블 모델
---
    앙상블 모델의 패러미터는 모두 GridSearchCV를 이용해 찾아낸 최적의 값으로 설정되었습니다.
    컴파일이 지나치게 오래 걸리는 관계로 이 라이브러리에서 GridSearchCV를 사용하지는 않습니다.

    1. Random Forest Regression
    2. Adaboost Regression
    3. Gradient Boosting Regression
    4. XGBoost Regression

만든이
---
    사회학과 현우열, 2019
    제주대학교 컴퓨터공학전공 인공지능 수업
"""

# Linear algebra library
import numpy as np

# Dataframe manipulation library
import pandas as pd

# Data visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Linear models
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV

# Non-linear models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Ensemble models
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor # 아나콘다 3 환경에서 conda install -c conda-forge xgboost로 설치

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from subprocess import check_output

import warnings
warnings.filterwarnings('ignore')


class RegressorKit:
    # 생성자 Constructor
    # 클래스 내부에서 반복적으로 사용될 객체(Instance)의 선언 및 초기화
    # __init__(self): 괄호 안에 클래스 패러미터 지정 가능
    def __init__(self):
        self.dataset = 0


    # 데이터셋 불러오기
    def read(self, df):
        self.dataset = pd.read_csv(df)
        print("데이터셋 불러오기 성공:", self.dataset.columns.tolist()) # 데이터셋의 필드값 출력


    # 원하는 수 만큼 레코드 불러오기
    def show(self, record):
        print(self.dataset.head(record))

    
    # 특정 필드 제거. field는 1차원 배열이어야 함
    def drop(self, field):
        dataset = self.dataset.drop(field, axis=1, inplace=True)


    # 각 필드에 있는 고유의 리코드 값
    def unique_record(self):
        for column in self.dataset:
            print(column,':', self.dataset[column].nunique())


    def heatmap(self):
        """
        열지도를 생성합니다.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.dataset.corr(), annot=True, cmap='cubehelix_r')
        plt.show()

    
    def boxplot(self, axis_x, axis_y):
        """
        Boxplot을 생성합니다.
        """
        f, sub = plt.subplots(1, 1, figsize=(12.18,5))
        sns.boxplot(x=self.dataset[axis_x], y=self.dataset[axis_y], ax=sub)
        sub.set(xlabel=axis_x, ylabel=axis_y);
        plt.show()


    def pairplots(self, fields, cluster):
        """
        pairplot을 생성합니다. fields는 1차원 배열로 입력해야 합니다.
        """
        plt.figure(figsize=(10, 6))
        sns.plotting_context('notebook', font_scale=1.2)
        graph = sns.pairplot(self.dataset, vars=fields, hue=cluster, height=2)
        graph.set(xticklabels=[])
        plt.show()


    def plot3d(self, axis_x, axis_y, axis_z, bubble, cluster, hover):
        """
        오픈 소스 데이터 시각화 라이브러리인 plotly의 3d plot 모델입니다.

        Parameters
        ---
            axis_x, axis_y, axis_z: x, y, z축으로 쓰일 필드
            bubble: 각 데이터의 버블 크기를 결정할 필드
            cluster: 각 데이터의 색을 결정할 필드
            hover: 각 데이터에 마우스 포인터가 올라갈 경우 추가 정보로 표시될 필드
        
        Return values
        ---
            trainer_x, trainer_y, tester_x, tester_y
        """
        df = px.data.gapminder()
        fig = px.scatter_3d(self.dataset, x=axis_x, y=axis_y, z=axis_z, size=bubble, color=cluster,
                            hover_data=[hover])
        fig.update_layout(scene_zaxis_type="log")
        fig.show()


    def split(self, df, feed_x, target_y):
        """
        Cross Validation. 전체 데이터셋을 학습용과 시험용으로 분류합니다.
        이 교차 검증 알고리즘은 모든 평가 모델에 내장되어 있습니다.\n

        Parameters
        ---
            df: read()로부터 불러온 csv 파일이 저장된 객체
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
        
        Return values
        ---
            trainer_x, trainer_y, tester_x, tester_y
        """

        train, test = train_test_split(df, train_size=0.9, test_size=0.1)
        # 학습용 데이터 비율 75%, 테스트용 데이터 비율 25%로 나누는 것을 4-fold cross validation이라고 한다.
        # 가장 흔히 쓰이는 비율은 학습용 90%, 테스트용 10%이다. 10-fold corss validation에 해당한다.

        # 학습용 데이터셋을 리스트로 변환
        trainer_x = train[feed_x]
        trainer_y = train[target_y]
        # 시험용 데이터셋을 리스트로 변환
        tester_x = test[feed_x]
        tester_y = test[target_y]

        return trainer_x, trainer_y, tester_x, tester_y


    def linear(self, feed_x, target_y):
        """
        Linear Regression

        Parameters
        ---
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(self.dataset, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        run = LinearRegression() # 알고리즘 구동 변수 선언
        run.fit(trainer_x, trainer_y) # 알고리즘을 이용한 학습
        predictor_y = run.predict(tester_x) # 테스트용 데이터 입력 후 예측

        R_squared = r2_score(tester_y, predictor_y) # 채택된 선형 모델의 결정계수 "R²"
        # R_squared = np.round(run.score(tester_x, tester_y) * 100, 5)

        print("-----------------")
        print("Linear Regression")
        print("-----------------")
        print("실제값:\n", tester_y.head(10))
        print("예측값:\n", predictor_y[:10])
        print("R²: %.3f" % R_squared)


    def ridge(self, feed_x, target_y):
        """
        Ridge Regression

        Parameters
        ---
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력

        Source
        ---
            https://bit.ly/2XjDZiV
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(self.dataset, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        alphas = 10 ** np.linspace(10, -2, 100) * 0.5

        # 최적의 패널티 탐색
        runCV = RidgeCV(alphas = alphas, cv=10, scoring = "neg_mean_squared_error", normalize = True) # 패러미터에 "cv = n"을 추가해 n-fold CV를 수행 / 기본적으로 LOOCV 수행
        runCV.fit(trainer_x, trainer_y)
        penalty = runCV.alpha_ # 최적의 λ 패널티

        run = Ridge(alpha = penalty, normalize = True) # 알고리즘 구동 변수 선언 / 최소제곱선에 패널티 부과
        run.fit(trainer_x, trainer_y) # 알고리즘을 이용한 학습
        predictor_y = run.predict(tester_x) # 테스트용 데이터 입력 후 예측

        R_squared = r2_score(tester_y, predictor_y) # 채택된 선형 모델의 결정계수 "R²"
        # R_squared = np.round(run.score(tester_x, tester_y) * 100, 5)

        print("----------------")
        print("Ridge Regression")
        print("----------------")
        print("실제값:\n", tester_y.head(10))
        print("예측값:\n", predictor_y[:10])
        print("R²: %.3f" % R_squared)

    
    def lasso(self, feed_x, target_y):
        """
        Lasso Regression

        Parameters
        ---
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력

        Source
        ---
            https://bit.ly/2XjDZiV
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(self.dataset, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        # 최적의 패널티 탐색
        runCV = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
        runCV.fit(trainer_x, trainer_y)
        penalty = runCV.alpha_ # 최적의 λ 패널티

        run = Lasso(max_iter = 100000, normalize = True) # 알고리즘 구동 변수 선언
        run.set_params(alpha = penalty) # 최소제곱선에 패널티 부과
        run.fit(trainer_x, trainer_y) # 알고리즘을 이용한 학습
        predictor_y = run.predict(tester_x) # 테스트용 데이터 입력 후 예측

        R_squared = r2_score(tester_y, predictor_y) # 채택된 선형 모델의 결정계수 "R²"
        # R_squared = np.round(run.score(tester_x, tester_y) * 100, 5)

        print("----------------")
        print("Lasso Regression")
        print("----------------")
        print("실제값:\n", tester_y.head(10))
        print("예측값:\n", predictor_y[:10])
        print("R²: %.3f" % R_squared)

    
    def elastic_net(self, feed_x, target_y):
        """
        Elastic-net Regression

        Parameters
        ---
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력

        Source
        ---
            https://bit.ly/2rQWPCv
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(self.dataset, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        # 최적의 패널티 탐색
        runCV = ElasticNetCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
        runCV.fit(trainer_x, trainer_y)
        penalty = runCV.alpha_ # 최적의 λ 패널티

        run = ElasticNet(alpha = penalty, max_iter = 100000)
        run.set_params(alpha = penalty) # 최소제곱선에 패널티 부과
        run.fit(trainer_x, trainer_y)
        predictor_y = run.predict(tester_x)

        R_squared = r2_score(tester_y, predictor_y) # 채택된 선형 모델의 결정계수 "R²"
        # R_squared = np.round(run.score(tester_x, tester_y) * 100, 5)

        print("----------------------")
        print("Elastic-net Regression")
        print("----------------------")
        print("실제값:\n", tester_y.head(10))
        print("예측값:\n", predictor_y[:10])
        print("R²: %.3f" % R_squared)


    def KNN(self, feed_x, target_y):
        """
        K-Nearest Neighbors Regression

        Parameters
        ---
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(self.dataset, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        run = KNeighborsRegressor(n_neighbors=10)
        run.fit(trainer_x, trainer_y)
        predictor_y = run.predict(tester_x)

        R_squared = r2_score(tester_y, predictor_y)

        print("--------------")
        print("KNN Regression")
        print("--------------")
        print("실제값:\n", tester_y.head(10))
        print("예측값:\n", predictor_y[:10])
        print("R²: %.3f" % R_squared)


    def SVM(self, feed_x, target_y):
        """
        RBF 커널을 이용한 Support Vector Machine Regression

        Parameters
        ---
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(self.dataset, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        run = SVR(kernel='rbf')
        run.fit(trainer_x, trainer_y)
        predictor_y = run.predict(tester_x)

        R_squared = r2_score(tester_y, predictor_y)

        print("--------------")
        print("SVM Regression")
        print("--------------")
        print("실제값:\n", tester_y.head(10))
        print("예측값:\n", predictor_y[:10])
        print("R²: %.3f" % R_squared)


    def decision_tree(self, feed_x, target_y):
        """
        Decision Tree Regression

        Parameters
        ---
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(self.dataset, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        run = DecisionTreeRegressor()
        run.fit(trainer_x, trainer_y)
        predictor_y = run.predict(tester_x)

        R_squared = r2_score(tester_y, predictor_y)

        print("------------------------")
        print("Decision Tree Regression")
        print("------------------------")
        print("실제값:\n", tester_y.head(10))
        print("예측값:\n", predictor_y[:10])
        print("R²: %.3f" % R_squared)


    def random_forest(self, feed_x, target_y):
        """
        Random Forest Regression

        Parameters
        ---
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(self.dataset, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        run = RandomForestRegressor(n_estimators=100) # 알고리즘 구동 변수 선언 / n_estimators: 만들어질 하위 나무의 개수
        run.fit(trainer_x, trainer_y) # 알고리즘을 이용한 학습
        predictor_y = run.predict(tester_x) # 테스트용 데이터 입력 후 예측

        R_squared = r2_score(tester_y, predictor_y)

        print("------------------------")
        print("Random Forest Regression")
        print("------------------------")
        print("실제값:\n", tester_y.head(10))
        print("예측값:\n", predictor_y[:10])
        print("R²: %.3f" % R_squared)


    def adaboost(self, feed_x, target_y):
        """
        Adaptive Boosting Regression

        Parameters
        ---
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(self.dataset, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        run = AdaBoostRegressor(base_estimator=None, n_estimators=500, learning_rate=0.5, random_state=1) # GridSearchCV를 이용해 찾아낸 최적의 패러미터 / base_estimator: 만들어질 하위 나무의 종류. None일 경우 Stump형 결정 나무가 이용됨
        run.fit(trainer_x, trainer_y) # 알고리즘을 이용한 학습
        predictor_y = run.predict(tester_x) # 테스트용 데이터 입력 후 예측

        R_squared = r2_score(tester_y, predictor_y)

        print("-------------------")
        print("Adaboost Regression")
        print("-------------------")
        print("실제값:\n", tester_y.head(10))
        print("예측값:\n", predictor_y[:10])
        print("R²: %.3f" % R_squared)


    def gradient_boost(self, feed_x, target_y):
        """
        Gradient Boosting Regression

        Parameters
        ---
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(self.dataset, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        run = GradientBoostingRegressor(n_estimators=1000, max_depth=4, min_samples_leaf=2, learning_rate=0.1) # GridSearchCV를 이용해 찾아낸 최적의 패러미터 
        run.fit(trainer_x, trainer_y) # 알고리즘을 이용한 학습
        predictor_y = run.predict(tester_x) # 테스트용 데이터 입력 후 예측

        R_squared = r2_score(tester_y, predictor_y)

        print("----------------------------")
        print("Gradient Boosting Regression")
        print("----------------------------")
        print("실제값:\n", tester_y.head(10))
        print("예측값:\n", predictor_y[:10])
        print("R²: %.3f" % R_squared)


    def xgboost(self, feed_x, target_y):
        """
        Extreme Gradient Boosting Regression

        Parameters
        ---
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(self.dataset, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        run = XGBRegressor(objective='reg:squarederror', n_estimators=1000, colsample_bytree=0.4, max_depth=2, learning_rate=0.1) # GridSearchCV를 이용해 찾아낸 최적의 패러미터 
        run.fit(trainer_x, trainer_y) # 알고리즘을 이용한 학습
        predictor_y = run.predict(tester_x) # 테스트용 데이터 입력 후 예측

        R_squared = r2_score(tester_y, predictor_y)

        print("------------------")
        print("XGBoost Regression")
        print("------------------")
        print("실제값:\n", tester_y.head(10))
        print("예측값:\n", predictor_y[:10])
        print("R²: %.3f" % R_squared)


    def execute(self, feed_x, target_y):
        self.linear(feed_x, target_y)
        self.ridge(feed_x, target_y)
        self.lasso(feed_x, target_y)
        self.elastic_net(feed_x, target_y)

        self.KNN(feed_x, target_y)
        self.SVM(feed_x, target_y)
        self.decision_tree(feed_x, target_y)
        
        self.random_forest(feed_x, target_y)
        self.adaboost(feed_x, target_y)
        self.gradient_boost(feed_x, target_y)
        self.xgboost(feed_x, target_y)