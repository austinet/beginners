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
        for key in params.keys():
            params[key] -= self.lr * grads[key]


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
            self.velocity = {}
            
            for key, val in params.items():
                self.velocity[key] = np.zeros_like(val)

        for key in params.keys():
            self.velocity[key] = (self.momentum * self.velocity[key]) - (self.lr * grads[key])
            params[key] += self.velocity[key]
        

class Nesterov:

    ''' Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901) '''
    # NAG: the improved momentum method (http://newsight.tistory.com/224)
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


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
            self.h = {}
            
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


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
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
        

class RMSprop:

    ''' RMSprop '''

    def __init__(self, learning_rate=0.01, decay_rate = 0.99):
        self.lr = learning_rate
        self.decay_rate = decay_rate
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)