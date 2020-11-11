"""
Danakit: Sample library of machine learning methods

Classes
---
    regressor
    classifier
    ensemble

Author
---
    Austin, 2019
    Dept. of Computer Engineering, Jeju National University

"""

# Linear algebra library
import numpy as np

# Dataframe manipulation library
import pandas as pd

# Regression models
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

# Ensemble models
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier

# Else..
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from subprocess import check_output
import matplotlib.pyplot as plt


class preprocess:
    """
    하위 클래스에 데이터 전처리 기능을 제공합니다.
    """


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


    def commaless(self, df):
        """
        쉼표(,)없는 리스트 출력 포멧 만들기

        Parameter
        ---
            df: 쉼표 없는 리스트로 바꿀 데이터셋

        Process
        ---
            1. 데이터셋을 2차원 배열로 반환
            2. 2차원 배열을 1차원 배열로 반환
            3. 1차원 배열을 문자열로 반환
            4. 문자열 내 쉼표 제거
        """

        multi_list = df.values.T.tolist() # 데이터셋을 2차원 배열로 전환: csv 데이터셋은 2차원 배열로 표현된다.
        
        uni_list = []
        for i in range(len(df.columns)): # i를 데이터셋에 포함된 필드 수 만큼 반복
            for j in range(len(df)): # j를 데이터셋의 자료 수 만큼 반복
                uni_list.append(multi_list[i][j]) # 2차원 배열의 자료를 새로운 1차원 배열에 저장

        a_str = str(uni_list) # 1차원 배열을 문자열(string)로 변환
        a_str = a_str.replace(",", "") # 문자열 내 쉼표(,) 제거

        return a_str


class regressor(preprocess):
    """
    Introduction
    ---
        데이터 예측 알고리즘을 제공합니다.
        regressor()로 초기화하는 것을 권장합니다.

    Regressors
    ---
        linear()
        ridge()
        lasso()
        elastic_net()
    """


    def linear(self, df, feed_x, target_y):
        """
        Linear Regression

        Parameters
        ---
            df: read()로부터 불러온 csv 파일이 저장된 객체
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(df, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        run = LinearRegression() # 알고리즘 구동 변수 선언
        run.fit(trainer_x, trainer_y) # 알고리즘을 이용한 학습
        predictor_y = run.predict(tester_x) # 테스트용 데이터 입력 후 예측

        y_intercept = run.intercept_ # 채택된 선형 모델의 y축 절편
        slope = run.coef_ # 채택된 선형 모델의 기울기. 기울기는 계수(Coefficient)에 해당한다.
        mse = mean_squared_error(tester_y, predictor_y) # 채택된 선형 모델의 평균제곱오차
        # mse = np.square(np.subtract(tester_y, predictor_y)).mean()
        R_squared = r2_score(tester_y, predictor_y) # 채택된 선형 모델의 결정계수 "R²"
        # R_squared = np.round(run.score(tester_x, tester_y) * 100, 5)

        print("-----------------")
        print("Linear Regression")
        print("-----------------")
        # The least sqaures fit
        print("slope:", slope)
        print("y intercept:", y_intercept)
        # Model evaluation
        print("실제값:\n", tester_y.head(10))
        print("예측값:\n", predictor_y[:10])
        print("Mean Squared Error: %.3f" % mse)
        print("R²: %.3f" % R_squared)


    def ridge(self, df, feed_x, target_y):
        """
        Ridge Regression

        Parameters
        ---
            df: read()로부터 불러온 csv 파일이 저장된 객체
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력

        Source
        ---
            https://bit.ly/2XjDZiV
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(df, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        # Declare list of possible penalty values
        # Penalty "alpha" is also called as "λ(lamda)"
        alphas = 10 ** np.linspace(10, -2, 100) * 0.5

        # Find the optimized λ penalty through cross validation
        runCV = RidgeCV(alphas = alphas, cv=10, scoring = "neg_mean_squared_error", normalize = True) # 패러미터에 "cv = n"을 추가해 n-fold CV를 수행 / 기본적으로 LOOCV 수행
        runCV.fit(trainer_x, trainer_y)
        penalty = runCV.alpha_ # 최적의 λ 패널티

        run = Ridge(alpha = penalty, normalize = True) # 알고리즘 구동 변수 선언 / 최소제곱선에 패널티 부과
        run.fit(trainer_x, trainer_y) # 알고리즘을 이용한 학습
        predictor_y = run.predict(tester_x) # 테스트용 데이터 입력 후 예측

        y_intercept = run.intercept_ # 채택된 선형 모델의 y축 절편
        slope = run.coef_ # 채택된 선형 모델의 기울기. 기울기는 계수(Coefficient)에 해당한다.
        mse = mean_squared_error(tester_y, predictor_y) # 채택된 선형 모델의 평균제곱오차
        # mse = np.square(np.subtract(tester_y, predictor_y)).mean()
        R_squared = r2_score(tester_y, predictor_y) # 채택된 선형 모델의 결정계수 "R²"
        # R_squared = np.round(run.score(tester_x, tester_y) * 100, 5)

        print("----------------")
        print("Ridge Regression")
        print("----------------")
        # The least sqaures fit with Ridge Penalty
        print("slope:", slope)
        print("y intercept:", y_intercept)
        print("Penalty(λ):", penalty)
        # Model evaluation
        print("실제값:\n", tester_y.head(10))
        print("예측값:\n", predictor_y[:10])
        print("Mean Squared Error: %.3f" % mse)
        print("R²: %.3f" % R_squared)

    
    def lasso(self, df, feed_x, target_y):
        """
        Lasso Regression

        Parameters
        ---
            df: read()로부터 불러온 csv 파일이 저장된 객체
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력

        Source
        ---
            https://bit.ly/2XjDZiV
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(df, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        # Find the optimized λ penalty through cross validation
        # Penalty "alpha" is also called as "λ(lamda)"
        runCV = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
        runCV.fit(trainer_x, trainer_y)
        penalty = runCV.alpha_ # 최적의 λ 패널티

        run = Lasso(max_iter = 100000, normalize = True) # 알고리즘 구동 변수 선언
        run.set_params(alpha = penalty) # 최소제곱선에 패널티 부과
        run.fit(trainer_x, trainer_y) # 알고리즘을 이용한 학습
        predictor_y = run.predict(tester_x) # 테스트용 데이터 입력 후 예측

        y_intercept = run.intercept_ # 채택된 선형 모델의 y축 절편
        slope = run.coef_ # 채택된 선형 모델의 기울기. 기울기는 계수(Coefficient)에 해당한다.
        mse = mean_squared_error(tester_y, predictor_y) # 채택된 선형 모델의 평균제곱오차
        # mse = np.square(np.subtract(tester_y, predictor_y)).mean()
        R_squared = r2_score(tester_y, predictor_y) # 채택된 선형 모델의 결정계수 "R²"
        # R_squared = np.round(run.score(tester_x, tester_y) * 100, 5)

        print("----------------")
        print("Lasso Regression")
        print("----------------")
        # The least sqaures fit with Lasso Penalty
        print("slope:", slope)
        print("y intercept:", y_intercept)
        print("Penalty(λ):", penalty)
        # Model evaluation
        print("실제값:\n", tester_y.head(10))
        print("예측값:\n", predictor_y[:10])
        print("Mean Squared Error: %.3f" % mse)
        print("R²: %.3f" % R_squared)

    
    def elastic_net(self, df, feed_x, target_y):
        """
        Elastic-net Regression

        Parameters
        ---
            df: read()로부터 불러온 csv 파일이 저장된 객체
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력

        Source
        ---
            https://bit.ly/2rQWPCv
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(df, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        # Find the optimized λ penalty through cross validation
        # Penalty "alpha" is also called as "λ(lamda)"
        runCV = ElasticNetCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
        runCV.fit(trainer_x, trainer_y)
        penalty = runCV.alpha_ # 최적의 λ 패널티

        run = ElasticNet(alpha = penalty, max_iter = 100000)
        run.set_params(alpha = penalty) # 최소제곱선에 패널티 부과
        run.fit(trainer_x, trainer_y)
        predictor_y = run.predict(tester_x)

        y_intercept = run.intercept_ # 채택된 선형 모델의 y축 절편
        slope = run.coef_ # 채택된 선형 모델의 기울기. 기울기는 계수(Coefficient)에 해당한다.
        mse = mean_squared_error(tester_y, predictor_y) # 채택된 선형 모델의 평균제곱오차
        # mse = np.square(np.subtract(tester_y, predictor_y)).mean()
        R_squared = r2_score(tester_y, predictor_y) # 채택된 선형 모델의 결정계수 "R²"
        # R_squared = np.round(run.score(tester_x, tester_y) * 100, 5)

        print("----------------------")
        print("Elastic-net Regression")
        print("----------------------")
        # The least sqaures fit with Elastic-net Penalty
        print("Slope:", slope)
        print("y intercept:", y_intercept)
        print("Penalty(λ):", penalty)
        print()
        print("실제값:\n", tester_y.head(10))
        print("예측값:\n", predictor_y[:10])
        print("Mean Squared Error: %.3f" % mse)
        print("R²: %.3f" % R_squared)


class classifier(preprocess):
    """
    Introduction
    ---
        데이터 분류 알고리즘을 제공합니다.
        classifier()로 초기화하는 것을 권장합니다.

    Classifiers
    ---
        logistic_regression()
        support_vector_machines()
        k_nearest_neighbors()
        decision_trees()
        naive_bayes()
    """


    def logistic_regression(self, df, feed_x, target_y):
        """
        Logistic Regression

        Parameters
        ---
            df: read()로부터 불러온 csv 파일이 저장된 객체
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(df, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        run = LogisticRegression() # 알고리즘 구동 변수 선언
        run.fit(trainer_x, trainer_y) # 알고리즘을 이용한 학습
        predictor_y = run.predict(tester_x) # 테스트용 데이터 입력 후 예측

        print("-------------------")
        print("Logistic Regression")
        print("-------------------")
        print("실제값:", tester_y)
        print("예측값:", predictor_y)
        print('정확도:', metrics.accuracy_score(predictor_y, tester_y) * 100)


    def support_vector_machines(self, df, feed_x, target_y):
        """
        Support Vector Machines

        Parameters
        ---
            df: read()로부터 불러온 csv 파일이 저장된 객체
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(df, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        run = svm.SVC() # 알고리즘 구동 변수 선언
        run.fit(trainer_x, trainer_y) # 알고리즘을 이용한 학습
        predictor_y = run.predict(tester_x) # 테스트용 데이터 입력 후 예측

        print("-----------------------")
        print("Support Vector Machines")
        print("-----------------------")
        print("실제값:", tester_y)
        print("예측값:", predictor_y)
        print('정확도:', metrics.accuracy_score(predictor_y, tester_y) * 100)


    def k_nearest_neighbors(self, df, feed_x, target_y):
        """
        k만큼 가까운 이웃 맺기 알고리즘

        Parameters
        ---
            df: read()로부터 불러온 csv 파일이 저장된 객체
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(df, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        print("-------------------")
        print("K-Nearest Neighbors")
        print("-------------------")

        neighbors = list(range(1, 21))
        accuracies = pd.Series() # k가 1 ~ 20일때 정확도를 list로 저장

        for k in neighbors:
            run = KNeighborsClassifier(k) # 알고리즘 구동 변수 선언
            run.fit(trainer_x, trainer_y) # 알고리즘을 이용한 학습
            predictor_y = run.predict(tester_x) # 테스트용 데이터 입력 후 예측

            print('k가', k, '일때 정확도:', metrics.accuracy_score(predictor_y, tester_y) * 100)

            accuracies = accuracies.append(pd.Series(metrics.accuracy_score(predictor_y, tester_y))) # k가 1 ~ 20일때 정확도를 그래프로 그리기 위한 좌표 저장

        # 정확도를 그래프로 나타내기
        plt.plot(neighbors, accuracies)
        plt.xticks(neighbors)
        plt.show()


    def decision_trees(self, df, feed_x, target_y):
        """
        결정 나무 알고리즘

        Parameters
        ---
            df: read()로부터 불러온 csv 파일이 저장된 객체
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(df, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        run = DecisionTreeClassifier() # 알고리즘 구동 변수 선언
        run.fit(trainer_x, trainer_y) # 알고리즘을 이용한 학습
        predictor_y = run.predict(tester_x) # 테스트용 데이터 입력 후 예측

        print("--------------")
        print("Decision Trees")
        print("--------------")
        print("실제값:", tester_y)
        print("예측값:", predictor_y)
        print('정확도:', metrics.accuracy_score(predictor_y, tester_y) * 100)


class ensemble(preprocess):
    """
    Introduction
    ---
        앙상블 알고리즘을 제공합니다.
        ensemble()로 초기화하는 것을 권장합니다.

    Classifiers
    ---
        random_forests()
        ada_boost()
        gradient_boost()
        xgboost()
    """

    def random_forest_regressor(self, df, feed_x, target_y):
        """
        무작위 숲 예측

        Parameters
        ---
            df: read()로부터 불러온 csv 파일이 저장된 객체
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(df, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        run = RandomForestRegressor(n_estimators=100) # 알고리즘 구동 변수 선언 / n_estimators: 만들어질 하위 나무의 개수
        run.fit(trainer_x, trainer_y) # 알고리즘을 이용한 학습
        predictor_y = run.predict(tester_x) # 테스트용 데이터 입력 후 예측

        mse = mean_squared_error(tester_y, predictor_y) # 채택된 선형 모델의 평균제곱오차
        # mse = np.square(np.subtract(tester_y, predictor_y)).mean()
        R_squared = r2_score(tester_y, predictor_y) # 채택된 선형 모델의 결정계수 "R²"
        # R_squared = np.round(run.score(tester_x, tester_y) * 100, 5)

        print("-------------------------")
        print("Random Forest Regresssion")
        print("-------------------------")
        print("실제값:", tester_y.head(10))
        print("예측값:", predictor_y[:10])
        print("Mean Squared Error: %.3f" % mse)
        print("R²: %.3f" % R_squared)


    def random_forest_classifier(self, df, feed_x, target_y):
        """
        무작위 숲 분류

        Parameters
        ---
            df: read()로부터 불러온 csv 파일이 저장된 객체
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(df, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        run = RandomForestClassifier(n_estimators=100) # 알고리즘 구동 변수 선언 / n_estimators: 만들어질 하위 나무의 개수
        run.fit(trainer_x, trainer_y) # 알고리즘을 이용한 학습
        predictor_y = run.predict(tester_x) # 테스트용 데이터 입력 후 예측

        print("----------------------------")
        print("Random Forest Classification")
        print("----------------------------")
        print("실제값:", tester_y)
        print("예측값:", predictor_y)
        print('정확도:', metrics.accuracy_score(predictor_y, tester_y) * 100)


    def adaboost_regressor(self, df, feed_x, target_y):
        """
        Adaboost Regression

        Parameters
        ---
            df: read()로부터 불러온 csv 파일이 저장된 객체
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(df, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        run = AdaBoostRegressor(base_estimator = None, n_estimators = 500, learning_rate = 0.01, random_state = 1) # 알고리즘 구동 변수 선언 / n_estimators: 만들어질 하위 나무의 개수
        run.fit(trainer_x, trainer_y) # 알고리즘을 이용한 학습
        predictor_y = run.predict(tester_x) # 테스트용 데이터 입력 후 예측

        mse = mean_squared_error(tester_y, predictor_y) # 채택된 선형 모델의 평균제곱오차
        # mse = np.square(np.subtract(tester_y, predictor_y)).mean()
        R_squared = r2_score(tester_y, predictor_y) # 채택된 선형 모델의 결정계수 "R²"
        # R_squared = np.round(run.score(tester_x, tester_y) * 100, 5)

        print("-------------------")
        print("Adaboost Regression")
        print("-------------------")
        print("실제값:", tester_y.head(10))
        print("예측값:", predictor_y[:10])
        print("Mean Squared Error: %.3f" % mse)
        print("R²: %.3f" % R_squared)


    def adaboost_classifier(self, df, feed_x, target_y):
        """
        Adaboost Classification

        Parameters
        ---
            df: read()로부터 불러온 csv 파일이 저장된 객체
            feed_x: 학습에 쓰일 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
            target_y: 예측할 필드. ["필드 1", "필드 2", ...]와 같이 list 입력
        """

        trainer_x, trainer_y, tester_x, tester_y = self.split(df, feed_x, target_y) # 교차 검증을 위해 데이터셋을 학습용, 시험용으로 분할

        run = AdaBoostClassifier(base_estimator = None, n_estimators = 100) # 알고리즘 구동 변수 선언 / base_estimator: 만들어질 weak learners의 종류. None일 경우 Stump형 결정 나무가 사용된다.
        run.fit(trainer_x, trainer_y) # 알고리즘을 이용한 학습
        predictor_y = run.predict(tester_x) # 테스트용 데이터 입력 후 예측

        print("-----------------------")
        print("Adaboost Classification")
        print("-----------------------")
        print("실제값:", tester_y)
        print("예측값:", predictor_y)
        print('정확도:', metrics.accuracy_score(predictor_y, tester_y) * 100)
