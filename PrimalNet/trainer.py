# trainer :: neural net training module #
# Based on Numpy #
# Austin Hyeon, 2020. "Sociology meets Computer Science" #

import sys, os
sys.path.append(os.pardir)
sys.path.insert(0, '/Users/austin/OneDrive/Repositories/MNIST/') # for macOS
sys.path.insert(0, '/Users/milim/Documents/GitHub/MNIST') # for Windows

import numpy as np
from netkit import MLP
from optimakit import *
from mnist import load_mnist


# load the dataset
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# initialize the network
network = MLP(input_nodes=784, hidden_nodes=[400, 400], output_nodes=10,
                activ='relu', weight_init_std='he', use_batch_norm=True, use_dropout=True)
optimizer = SGD(learning_rate=0.1)

# set hyperparameters
interations = 10000
train_size = x_train.shape[0]
batch_size = 100
iters_per_epoch = max(train_size / batch_size, 1)

# set lists for recording results
train_losses = []
train_accuracies = []
test_accuracies = []

print('accuracy to the training set | accuracy to the test set')

for i in range(interations):
    # create batches
    bnum = np.random.choice(train_size, batch_size) # Volume of a batch
    x_batch = x_train[bnum] # training batch
    t_batch = t_train[bnum] # test batch

    # settings for optimization
    grads = network.echo(x_batch, t_batch)
    params = network.params

    # optimizing the parameters: weigths & biases
    optimizer.update(params, grads)

    loss = network.get_loss(x_batch, t_batch) # calcuate residuals(losses) between x_batch and t_batch
    train_losses.append(loss) # record losses

    # cross validation: get accuracies
    if i % iters_per_epoch == 0:
        '''
        train_acc: accuracy to the training sets
        test_acc: accuracy to the test sets

        < bias-variance trade off >
        overfitting: high train_acc, low test_acc
        underfitting: low train_acc, high test_acc
        '''
        train_acc = network.get_accuracy(x_train, t_train)
        test_acc = network.get_accuracy(x_test, t_test)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    
        print('{} | {:10.4f} | {}'.format(i, train_acc, test_acc))