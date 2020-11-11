'''
Deep Learning:
Linear Regression Algorithm
Logistic Regression Algorithm

Keywords:
Hypothesis, H(x) (1차 함수)
Cost Function, cost(W) (2차 함수)
Gradient Descent Algorithm (미분)
Matrix (행렬)

Source codes form https://youtu.be/mQGwjrStQgg
'''

import tensorflow as tf
import matplotlib.pyplot as plt


# 선형 회귀 알고리즘
def LineRegr(xdata, ydata, loop):
    # xdata 및 ydata가 들어갈 공간 선언
    # shape=[a, b]는 세로 a개, 가로 b개인 행렬을 만들겠다는 선언
    # 이 경우 몇 개의 xdata가 들어올 지 결정되지 않았으므로 None 배정
    # 자세한 설명은 다음 참고: https://youtu.be/kPxpJY6fRkY
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    # 가설 함수 H(x) = Wx + b에 필요한 기울기 W, y축 가중치 b 무작위 생성
    W = tf.Variable(tf.random_normal([3, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # 가설 행렬 H(x) = XW 정의
    hypothesis = tf.matmul(X, W) + b
    # 텐서플로우를 이용해 무작위로 생성된 가설 함수와 모든 데이터 간의 거리 평균값을 계산
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    # Gradient Descent Algorithm
    # 가장 작은 평균 거라값 w를 추적하는 알고리즘
    # 이때 필요한 그래프는 다음 함수 ShowCostGraph()에 있음
    # 설명은 다음을 참고: https://youtu.be/Y0EF9VqRuEA
    train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

    sess = tf.Session()
    # 그래프 내 변수 초기화(루프가 반복될 때마다 새로운 무작위 W, b값을 받아야하기 때문)
    sess.run(tf.global_variables_initializer())

    # 학습 알고리즘 n번 반복
    for i in range(loop):
        cost_val, h_val, _ = sess.run(
            [cost, hypothesis, train], feed_dict={X: xdata, Y: ydata}
        )
        if i % 10 == 0:
            print(i, "평균 거리값:", cost_val, "예측 결과값:", h_val)


# 로지스틱 회귀 알고리즘
def LogisRegr(xdata, ydata, loop):
    X = tf.placeholder(tf.float32, shape=[None, 2])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([2, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # 가설 시그모이드 함수의 정의
    # 시그모이드 함수식은 다음을 참고: https://youtu.be/PIjno6paszY
    hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

    # 최적의 시그모이드 함수를 찾기 위한 cost 함수의 정의
    # 모든 데이터와의 평균 거리값이 가장 작은 시그모이드 함수 경사도(W)를 찾는 것
    # cost 함수식은 다음을 참고: https://youtu.be/6vzchGYEJBc
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

    # Gradient Descent Algorithm
    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    # H(x)가 0.5 이하면 0, 이상이면 1로 환원
    # predicted에는 기계가 예측한 결과 0 또는 1이 저장됨
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    # 예측한 값 predicted가 실제 트레이닝 데이터 셋의 결과 Y와 동일한지 비교 후 정답률 계산
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(loop):
            cost_val, W_val, _ = sess.run([cost, W, train], feed_dict={X: xdata, Y: ydata})
            if i % 200 == 0:
                print(i, "평균 거리값:", cost_val, "시그모이드 경사도:", W_val)

        h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: xdata, Y: ydata})
        print("\nHypothesis:", h, "\nCorrect Y:", c, "\nAccuracy:", a)


'''
SingularLineRegr()은 선형 회귀 알고리즘의 개념 이해를 돕기 위해 입력값이 하나 뿐인 단순 모델이다.
SimpleLineRegr() 및 ShowCostGraph()는 cost(W)와 W에 대한 2차 함수 그래프를 보여주기 위해 작성되었다.
설명은 다음을 참고:
https://youtu.be/TxIVr-nk1so
https://youtu.be/Y0EF9VqRuEA
'''
# 입력 변수가 하나인 선형 회귀 알고리즘
def SingularLineRegr(xdata, ydata, loop):
    # 패러미터 설명:
    # x와 y은 2차원 그래프 상 데이터의 위치. loop는 학습 횟수

    x = tf.placeholder(tf.float32, shape=[None])
    y = tf.placeholder(tf.float32, shape=[None])
 
    # 가설 함수 H(x) = Wx + b에 필요한 무작위 W, b값 생성
    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # 무작위 가설 함수 H(x) = Wx + b 생성
    hypothesis = W * x + b

    # 텐서플로우를 이용해 무작위로 생성된 가설 함수와 모든 데이터 간의 거리 평균값을 계산
    cost = tf.reduce_mean(tf.square(hypothesis - y))

    # Gradient Descent Algorithm
    # 가장 작은 평균 거라값 w를 추적하는 알고리즘
    # 이때 필요한 그래프는 다음 함수 ShowCostGraph()에 있음
    # 설명은 다음을 참고: https://youtu.be/Y0EF9VqRuEA
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    # 그래프 내 변수 초기화(루프가 반복될 때마다 새로운 무작위 W, b값을 받아야하기 때문)
    sess.run(tf.global_variables_initializer())

    # 위의 학습 알고리즘을 n번 반복시킴
    for loop in range(1, 5001):
        cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
        # 처음 선언된 좌표값 공간에 실제 데이터 입력
        # 패러미터를 통해 외부로부터 값을 받아 옴
        feed_dict={x: xdata, y: ydata})
        # 학습 경과를 20번 반복마다 보여줌
        if loop % 20 == 0:
            print(loop, cost_val, W_val, b_val)


# y축 가중치 b가 없는 가설 함수 H(x) = Wx를 이용한 선형 회귀 알고리즘
def SimpleLineRegr(xline, yline):
    # 가설 함수 H(x) = Wx + b에 필요한 무작위 W, b값 생성
    W = tf.Variable(tf.random_normal([1]), name='weight')

    # 무작위 가설 함수 H(x) = Wx 생성
    hypothesis = W * xline

    # 텐서플로우를 이용해 무작위로 생성된 가설 함수와 모든 데이터 간의 거리 평균값을 계산
    cost = tf.reduce_mean(tf.square(hypothesis - yline))

    # Gradient Descent Algorithm
    # 가장 작은 평균 거라값 w를 추적하는 알고리즘
    # 이때 사용되는 그래프는 ShowCostGraph()에서 볼 수 있음
    # 설명은 다음을 참고: https://youtu.be/Y0EF9VqRuEA
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    # 그래프 내 변수 초기화(루프가 반복될 때마다 새로운 무작위 W, b값을 받아야하기 때문)
    sess.run(tf.global_variables_initializer())

    # 위의 학습 알고리즘을 n번 반복시킴
    for loop in range(1, 5001):
        cost_val, W_val, _ = sess.run([cost, W, train])
        # 학습 경과를 20번 반복마다 보여줌
        if loop % 20 == 0:
           print(loop, cost_val, W_val)


# x축이 W이고 y축이 cost(W)인 2차원 그래프 그리기
def ShowCostGraph(xline, yline):
    # 가설 함수 H(x) = Wx에 필요한 무작위 W값 생성
    W = tf.placeholder(tf.float32)

    # 무작위 가설 함수 H(x) = Wx 생성(단순화 됨)
    hypothesis = W * xline

    # 텐서플로우를 이용해 무작위로 생성된 가설 함수와 모든 데이터 간의 거리 평균값을 계산
    cost = tf.reduce_mean(tf.square(hypothesis - yline))
    
    sess = tf.Session()
    # 그래프 내 변수 초기화(루프가 반복될 때마다 새로운 무작위 W값을 받아야하기 때문)
    sess.run(tf.global_variables_initializer())

    # Variables for plotting cost function
    W_val = []
    cost_val = []

    # 위의 학습 알고리즘을 n번 반복시킴
    for i in range(-30, 50):
        feed_W = i * 0.1
        curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
        W_val.append(curr_W)
        cost_val.append(curr_cost)
        print("x축:", feed_W, "y축:", curr_cost)

    # Show the cost function
    plt.plot(W_val, cost_val)
    plt.show()

