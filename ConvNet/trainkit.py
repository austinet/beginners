# trainkit :: neural network training module #
# Based on Numpy #
# Austin Hyeon, 2020. "Sociology meets Computer Science" #

import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from optimakit import *

class Trainer:
    
    ''' the trainer class for the neural nets '''

    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):

        ''' set variables '''

        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        ''' set optimizer '''

        optimizer_switcher = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprop':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_switcher[optimizer.lower()](**optimizer_param)

        ''' set the numbers of learning '''
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        ''' lists for archiving results '''

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []


    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        ''' activate the net '''

        grads = self.network.echo(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        ''' get losses '''
        
        loss = self.network.get_loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        # if self.verbose: print("train loss:" + str(loss)) # print a loss per 1 echoing
        
        ''' calcuate & print out accuracies '''

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
                
            # cacluate
            train_acc = self.network.get_accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.get_accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            # print out
            if self.verbose: print("epoch:" + str(self.current_epoch) + " | train_acc:" + str(train_acc) + " | test_acc:" + str(test_acc))
        self.current_iter += 1


    def run(self):

        ''' the main trigger of trainer class '''

        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.get_accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))

        ''' loss floating visualization '''

        plt.plot(range(self.max_iter), self.train_loss_list)
        plt.xlabel('epoch')
        plt.ylabel('loss')

        plt.show()