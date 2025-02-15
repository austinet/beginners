# optimakit :: optimizing methods module #
# Based on Numpy #
# Austin Hyeon, 2020. "Sociology meets Computer Science" #

import numpy as np

class SGD:

    ''' Stochastic Gradient Descent '''

    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        

    def update(self, params, grads):
        '''
        parameters:
        ---
            params: this can be found in the network classes.
            grads: a variable to access network.echo(x, t).
        '''
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class Momentum:

    ''' Momentum Optimizer '''

    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads):
        '''
        parameters:
        ---
            params: this can be found in the network classes.
            grads: a variable to access network.echo(x, t).
        '''
        if self.velocity is None:
            self.velocity = []
            
            for param in params:
                self.velocity.append(np.zeros_like(param))

        for i in range(len(params)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grads[i]
            params[i] += self.velocity[i]
        

class Nesterov:

    ''' Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901) '''
    # NAG: the improved momentum method (http://newsight.tistory.com/224)
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))
            
        for i in range(len(params)):
            self.v[i] *= self.momentum
            self.v[i] -= self.lr * grads[i]
            params[i] += self.momentum * self.momentum * self.v[i]
            params[i] -= (1 + self.momentum) * self.lr * grads[i]


class AdaGrad:

    ''' Adaptive Gradient '''

    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.h = None

    def update(self, params, grads):
        '''
        parameters:
        ---
            params: this can be found in the network classes.
            grads: a variable to access network.echo(x, t).
        '''
        if self.h is None:
            self.h = []
            
            for param in params:
                self.h.append(np.zeros_like(param))

        for i in range(len(params)):
            self.h[i] += grads[i] * grads[i]
            params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)


class Adam:

    ''' Adaptive Moment Estimation (http://arxiv.org/abs/1412.6980v8) '''

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        '''
        parameters:
        ---
            params: this can be found in the network classes.
            grads: a variable to access network.echo(x, t).
        '''
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
        

class RMSprop:

    ''' RMSprop '''

    def __init__(self, learning_rate=0.01, decay_rate = 0.99):
        self.lr = learning_rate
        self.decay_rate = decay_rate
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))
            
        for i in range(len(params)):
            self.h[i] *= self.decay_rate
            self.h[i] += (1 - self.decay_rate) * grads[i] * grads[i]
            params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)