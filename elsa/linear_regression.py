# Simple Linear Regression on Tensorflow 2.0 #
# Austin, 2019 #

import tensorflow as tf
import numpy as np


tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

'''
config.threading
   ---
    set_intra_op_parallelism_threads(n)
    Set number of threads used within an individual operation for parallelism.
    하나의 연산작업(an individual operation)의 내부(intra)에서 사용될 thread 수를 지정한다.

    set_inter_op_parallelism_threads(n)
    Set number of threads used for parallelism between independent operations.
    다수의 서로 다른 연산작업(independent operations) 간(inter/between)에 사용될 thread 수를 지정한다.
'''


def simple():
    '''
    simple()
    ---
        변수(x축 값)가 하나인 1차함수 회귀
    '''
    # Dataset
    x_data = [1, 2, 3, 4, 5]
    y_data = [1, 2, 3, 4, 5]

    # Initialize the least squares fit's slope(W) and y-intercept(b)
    W = tf.Variable(tf.random.normal([1], mean=100, stddev=4))
    b = tf.Variable(tf.random.normal([1], mean=100, stddev=4))

    '''
    Variable()
    ---
        Tensorflow에서는 a = 0과 같이 변수를 선언, 초기화할 수 없다.
        따라서 a = tf.Variable(0)과 같이 선언한다.

    random.normal()
    ---
        가우시안 분포에서 무작위 값을 반환하는 함수.
        mean은 평균이며, 가우시안 분포의 중심 값을 말한다.
        stddev는 표준편차이며, 가우시안 분포에서 자료가 얼마나 분산되어 있는가를 나타낸다. 표준편차 값이 낮을수록 자료가 평균에 밀집해 있고, 그 반대인 경우 자료가 평균으로부터 분산되어 있다.
        배열 [] 안의 숫자는 반환될 무작위 값의 갯수이다. 예를 들어 [2, 2]일 경우 가로 2, 세로 2인 행렬에 총 네 개의 무작위 값이 반환된다.
    '''

    # Hyperparameter for gradient descent algorithm
    learning_rate = 0.01

    print("Epoch | Slope | y-Intercept | MSE(cost)")

    # Gradient descent algorithm
    for i in range(3001):
        with tf.GradientTape() as tape: # 루프에 따른 변수(W, b)를 tape에 기록
            hypothesis = W * x_data + b  # Define the least squares fit
            cost = tf.reduce_mean(tf.square(hypothesis - y_data)) # Define 'cost function' to get the mean squared error

            '''
            cost function
            ---
                비용 함수는 최소제곱선의 기울기 값(x축)과 최소제곱오차 값(y축) 사이의 관계를 나타낸 함수이다. 보통 U 모양의 포물선으로 나타난다.

            reduce_mean()
            ---
                reduce_mean()은 평균을 구하는 함수. 예를 들어 array = [1, 2, 3, 4]를 reduce_mean(array) 했을 경우, 결과는 2.5이다.
                https://webnautes.tistory.com/1235 참고.

            square()
            ---
                square()는 제곱연산 함수. 예를 들어 square(4)는 16이다.
            '''

        # Gradient
        W_gradient, b_gradient = tape.gradient(cost, [W, b])

        '''
        gradient descent
        ---
            경사 하강은 비용 함수를 미분하는 알고리즘이다. 이떄, 미분 간격을 조정하는 하이퍼패러미터를 learning rate라고 한다.

        tape.gradient(cost, [W, b])
        ---
            tape에 기록된 W, b값과 비용 함수식(cost)을 gradient()의 패러미터로 넘긴다.
            gradient()는 정의된 식(cost) 위에서 W, b에 대한 편미분을 진행, 그 결과를 각각 W_gradient, b_gradient에 넣는다.
        '''

        # Descent
        W.assign_sub(learning_rate * W_gradient)
        b.assign_sub(learning_rate * b_gradient)

        '''
        assign_sub()
        ---
            예를 들어, a.assign_sub(b)는 a = a - b와 같다.
            다음 연산을 위해 미분된 W, b에 learning rate를 반영한 뒤 다시 W, b 값에 넣는다.
            W, b 값을 포물선 밑부분을 향해 하강시키는 작업이다.
        '''

        if i % 10 == 0:
            # Epoch, Slope, y-Intercept, MSE(cost) 출력 
            print('{}|{}|{}|{:0.3f}'.format(i, W.numpy(), b.numpy(), cost))


    '''
    Gradient descent 분해하기
    ---
        앞서 gradient descent는 다음과 같은 코드로 구현되었다.

        # Gradient
        W_gradient, b_gradient = tape.gradient(cost, [W, b])

        # Descent
        W.assign_sub(learning_rate * W_gradient)
        b.assign_sub(learning_rate * b_gradient)

        이 코드의 내부는 다음과 같다. 기울기 W에 대해서만 미분을 진행할 경우:

        residual = tf.multiply(W, x_data) - y_data # H(x) - y에 해당: 최소제곱선과 데이터 사이의 거리
        gradient = tf.reduce_mean(tf.multiply(residual, x_data)) # 미분
        descent = W - tf.multiply(learning_rate, gradient) # 하강
        W.assign(descent) # 하강 후 W값을 저장
    '''


def multiple():
    '''
    multiple()
    ---
        변수(x축 값)가 여러 개인 1차함수 회귀
    '''

    # Dataset
    data = np.array([
        # X1   X2   X3   y
        [ 73.,  80.,  75., 152. ],
        [ 93.,  88.,  93., 185. ],
        [ 89.,  91.,  90., 180. ],
        [ 96.,  98., 100., 196. ],
        [ 73.,  66.,  70., 142. ]
    ], dtype=np.float32)

    # Slice the dataset into feeds and a target
    x_data = data[:, :-1]
    y_data = data[:, [-1]]

    # Initialize the least squares fit's slope(W) and y-intercept(b)
    W = tf.Variable(tf.random.normal([3, 1])) # 세로 3, 가로 1인 가중치 행렬 생성
    b = tf.Variable(tf.random.normal([1]))

    # Hyperparameter for gradient descent algorithm
    learning_rate = 0.0001

    '''
    learning rate
    ---
        learning rate가 너무 작으면 연산량이 지나치게 많아져 연산작업이 엉킬 수 있으며,
        반대로 너무 크면 미분 대상인 변수가 너무 빠르게 0으로 수렴하여 NaN 오류를 낸다.

        따라서 적절한 learning rate를 찾는 것이 deep learning의 중요한 요소 중 하나이다.
    '''

    # Define the least squares fit
    def hypothesis():
        return tf.matmul(x_data, W) + b

    '''
    matmul(a, b)
    행렬 a와 행렬 b의 곱셈. a와 b는 반드시 배열 []의 형태로 반환되어야 한다.
    '''

    # Gradient descent algorithm
    for i in range(10001):
        with tf.GradientTape() as tape: # 루프에 따른 변수(W, b)를 tape에 기록
            # hypothesis = W * x_data + b :: 변수가 여러 개이고 그에 따른 가중치 W 갯수도 늘어나기 때문에 위의 def hypothesis()로 대체
            cost = tf.reduce_mean(tf.square(hypothesis() - y_data)) # Define 'cost function' to get the mean squared error

        # Gradient
        W_gradient, b_gradient = tape.gradient(cost, [W, b])

        # Descent
        W.assign_sub(learning_rate * W_gradient)
        b.assign_sub(learning_rate * b_gradient)

        if i % 100 == 0:
            # Epoch, Slope, y-Intercept, MSE(cost) 출력 
            print('{} | {:10.3f}'.format(i, cost.numpy()))