# TFRegr :: Regression Algorithms with Tensorflow #
# Available on Tensorflow 2.0 #
# Austin Hyeon, 2019. "Sociology meets Computer Science" #

import tensorflow as tf

THREAD = tf.config.threading.get_inter_op_parallelism_threads()
tf.config.threading.set_inter_op_parallelism_threads(THREAD)

class linear:
    '''
    Linear Regression
    ---
        It provides the regression method using a linear function.

    Parameters
    ---
        feed_X: data for training machines
        target_y: the target data to predict
    '''
    def __init__(self, feed_X, target_y):
        # Class parameters
        self.feed_X = feed_X
        self.target_y = target_y
        
        # Initialize the least squares fit's slope(W) and y-intercept(b)
        fields = len(feed_X[0]) # Number of fields on axis x
        self.W = tf.Variable(tf.random.normal([fields, 1]))
        self.b = tf.Variable(tf.random.normal([1]))


    def _linear_function(self):
        '''
        Linear function "f(x)"
        '''
        fx = tf.matmul(self.feed_X, self.W) + self.b
        return fx


    def _cost_function(self, fx):
        '''
        Parameters
        ---
            fx: f(x). Linear function where to get the mean squared error
        '''
        MSE = tf.reduce_mean(tf.square(fx - self.target_y))
        return MSE


    def _gradient_descent(self, tape, cost, learning_rate):
        '''
        Gradient Descent Algorithm
        '''
        # Gradient
        W_gradient, b_gradient = tape.gradient(cost, [self.W, self.b])
        # Descent
        self.W.assign_sub(learning_rate * W_gradient)
        self.b.assign_sub(learning_rate * b_gradient)

        return self.W, self.b


    def regression(self, epoch, learning_rate):
        '''
        Parameters
        ---
            epoch: number of processing differentiation
            learning_rate: the rate of speed where the gradient moves during gradient descent
        '''

        for i in range(epoch+1):
            with tf.GradientTape() as tape:
                hypothesis = self._linear_function()
                cost = self._cost_function(hypothesis)

            self.W, self.b = self._gradient_descent(tape, cost, learning_rate)

            if i % 100 == 0:
                print('{} | {:0.3f} | {}'.format(i, cost, hypothesis))


class logistic:
    '''
    Logistic Regression
    ---
        It provides the binary classification method using a sigmoid function.

    Parameters
    ---
        feed_X: data for training machines
        target_y: the target data to predict
    '''
    def __init__(self, feed_X, target_y):
        # Class parameters
        self.feed_X = feed_X
        self.target_y = target_y

        # Initialize the least squares fit's slope(W) and y-intercept(b)
        fields = len(feed_X[0]) # Number of fields on axis x
        self.W = tf.Variable(tf.random.normal([fields, 1]))
        self.b = tf.Variable(tf.random.normal([1]))
    

    def _sigmoid_function(self):
        '''
        Sigmoid function "S(x)"
        '''
        sx = tf.divide(1., 1. + tf.exp(-tf.matmul(self.feed_X, self.W) + self.b))
        # sx = tf.nn.sigmoid(tf.matmul(self.feed_X, self.W) + self.b)
        return sx


    def _cost_function(self, sx): # also known as "Cross-Entropy"
        '''
        Parameters
        ---
            sx: S(x). Sigmoid function where to get the entropy value
        '''
        COST = -tf.reduce_mean(self.target_y * tf.math.log(sx) + (1 - self.target_y) * tf.math.log(1 - sx))
        return COST

    
    def _gradient_descent(self, tape, cost, learning_rate):
        '''
        Gradient Descent Algorithm
        '''
        # Gradient
        W_gradient, b_gradient = tape.gradient(cost, [self.W, self.b])
        # Descent
        self.W.assign_sub(learning_rate * W_gradient)
        self.b.assign_sub(learning_rate * b_gradient)

        return self.W, self.b


    def _cast_encode(self, sx):
        '''
        Cast Encoding
        '''
        CAST = tf.cast(sx > 0.5, dtype=tf.float32)
        return CAST


    def regression(self, epoch, learning_rate):
        '''
        Parameters
        ---
            epoch: number of processing differentiation
            learning_rate: the rate of speed where the gradient moves during gradient descent
        '''

        for i in range(epoch+1):
            with tf.GradientTape() as tape:
                hypothesis = self._sigmoid_function()
                cost = self._cost_function(hypothesis)
                predicted = self._cast_encode(hypothesis)

            self.W, self.b = self._gradient_descent(tape, cost, learning_rate)

            if i % 100 == 0:
                print('{} | {:0.3f} | {}'.format(i, cost, predicted))


class softmax:
    '''
    Softmax Regression
    ---
        It provides the multinomial classification method using a softmax function.
        Also known as multinomial logisitic regression, multiclass regression, and the maximum entropy classifier.

    Parameters
    ---
        feed_X: data for training machines
        target_y: the target data to predict
    '''
    def __init__(self, feed_X, target_y):
        # Class parameters
        self.feed_X = feed_X
        self.target_y = target_y

        # Initialize the least squares fit's slope(W) and y-intercept(b)
        fields = len(feed_X[0]) # Number of fields on axis x
        classes = len(target_y[0]) # Number of classes on axis y
        self.W = tf.Variable(tf.random.normal([fields, classes]))
        self.b = tf.Variable(tf.random.normal([classes]))

    
    def _softmax_function(self):
        '''
        Softmax function "S(z)"
        '''
        sz = tf.nn.softmax(tf.matmul(self.feed_X, self.W) + self.b)
        # logit = tf.matmul(self.feed_X, self.W) + self.b
        # sz = tf.exp(logit) / tf.reduce_sum(tf.exp(logit), axis = 0)
        return sz


    def _cost_function(self, softmax):
        '''
        Parameters
        ---
            sx: S(x). Softmax function where to get the cost value
        '''
        cost = -tf.reduce_sum(self.target_y * tf.math.log(softmax), axis=1)
        MEAN_COST = tf.reduce_mean(cost)

        return MEAN_COST


    def _gradient_descent(self, tape, cost, learning_rate):
        '''
        Gradient Descent Algorithm
        '''
        # Gradient
        W_gradient, b_gradient = tape.gradient(cost, [self.W, self.b])
        # Descent
        self.W.assign_sub(learning_rate * W_gradient)
        self.b.assign_sub(learning_rate * b_gradient)

        return self.W, self.b


    def _onehot_encode(self, sz):
        '''
        One-Hot Encoding
        '''
        onehot = tf.math.argmax(sz, axis=1)
        return onehot


    def regression(self, epoch, learning_rate):
        '''
        Parameters
        ---
            epoch: number of processing differentiation
            learning_rate: the rate of speed where the gradient moves during gradient descent
        '''
        for i in range(epoch+1):
            with tf.GradientTape() as tape:
                hypothesis = self._softmax_function()
                cost = self._cost_function(hypothesis)
                predicted = self._onehot_encode(hypothesis)

            self.W, self.b = self._gradient_descent(tape, cost, learning_rate)

            if i % 100 == 0:
                print('{} | {:0.3f}| {}'.format(i, cost, predicted))