import numpy as np # 수학 연산 수행을 위한 모듈
import pandas as pd # 데이터 처리를 위한 모듈
import seaborn as sns # 데이터 시각화 모듈
import matplotlib.pyplot as plt # 데이터 시각화 모듈

# 다양한 분류 알고리즘 패키지를 임포트함.
from sklearn.linear_model import LogisticRegression  # Logistic Regression 알고리즘
#from sklearn.cross_validation import train_test_split # 데이타 쪼개주는 모듈
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

#1) class 문 작성
#2) indentation
#3) self 파라미터 삽입

class ClassificationUtil: #gildong

    # 생성자 Constructor
    # 클래스 내부에서 반복적으로 사용될 객체(Instance)의 선언 및 초기화
    # __init__(self): 괄호 안에 클래스 패러미터 지정 가능
    def __init__(self):
        super().__init__()
        self.df = 0


    # csv 파일을 로드함. 예)df = read("a.csv")
    def read(self, dataset):
        self.df = pd.read_csv(dataset)

    
    def show(self):
        print(self.df.info())
        print(self.df.head(5))
        print(self.df.shape)

    def drop(self, col):
        # 불필요한 열(ID) 제거
        self.df.drop(col, axis=1, inplace=True)  # ID라는 컬럼(열)을 삭제하라는 의미

        # 불필요한 Id 컬럼 삭제
        # axis=1 : 컬럼을 의미
        # inplace=True : 삭제한 후 데이터 프레임에 반영하라는 의미

    # myplot(df, 'Height', 'Weight', 'Sex')
    def myplot(self, x_col, y_col, color_field):
        cl = self.df[color_field].unique()
        col = ['orange', 'blue', 'red', 'yellow', 'black', 'brown']

        fig = self.df[self.df[color_field] == cl[0]].plot(kind='scatter', x=x_col, y=y_col, color=col[0], label=cl[0])
        for i in range(len(cl)-1):
            self.df[self.df[color_field] == cl[i+1]].plot(kind='scatter', x=x_col, y=y_col, color=col[i+1], label=cl[i+1], ax=fig)

        fig.set_xlabel(x_col)
        fig.set_ylabel(y_col)
        fig.set_title(x_col + " vs. " + y_col)
        fig=plt.gcf()
        fig.set_size_inches(10, 8)
        plt.show()

    # myhist(df)
    def myhist(self, a):
        a.hist(edgecolor='black', linewidth=1.2)
        fig = plt.gcf()
        fig.set_size_inches(12,10)
        plt.show()


    #myviolinplot(df, 'Height', 'Weight')
    #a(성별, 학년)에 따라서 b(키, 몸무게)의 분포를 보여줌.
    #예) 성별에 따라 키의 분포를 보여줌.
    def myviolinplot(self, a, b):
        plt.figure(figsize=(5,4))
        plt.subplot(1,1,1)
        sns.violinplot(x=a,y=b,data=self.df)
        plt.show()


    #예)히트맵으로 성별과 가장 상관관계가 높은 필드(발크기,
    # 몸무게, 키 등)를 알 수 있음.
    def heatmap(self):
        plt.figure(figsize=(14, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='cubehelix_r')
        plt.show()


    #전체데이터를 학습용(70%), 테스트 용(30%)으로 나눔
    def data_nanum(self):
        a, b = train_test_split(self.df, test_size=0.3)
        # train=70% and test=30%
        return a, b


    #데이터프레임에서 원하는 컬럼만을 뽑아냄.
    # a = p.extract(df, 'Sex')
    # a = p.extract(df, ['Sex'])
    def extract(self, df, field_list):
        t = df[field_list]
        return t


    def ignore_warning(self):
        import warnings
        warnings.filterwarnings('ignore')

    #run_svm(['PetalLengthCm', 'SepalWidthCm'], 'Species')
    def run_svm(self, list, target):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[list]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[list]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = svm.SVC()  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        rate = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', rate)
        return rate


    def run_logistic_regression(self, list, target):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[list]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[list]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = LogisticRegression()  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        rate = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', rate)
        return rate


    def run_neighbor_classifier (self, list, target, num):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[list]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[list]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = KNeighborsClassifier(n_neighbors=num)  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        rate = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', rate)
        return rate


    def run_decision_tree_classifier (self, list, target):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[list]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[list]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = DecisionTreeClassifier()  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        rate = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', rate)
        return rate
