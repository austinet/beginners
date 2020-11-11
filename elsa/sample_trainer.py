import sys, os
sys.path.append(os.pardir)
sys.path.insert(0,'/Users/austin/OneDrive/Repositories/MNIST/')

import numpy as np
from sample_net import SampleNet
from mnist import load_mnist


# load the dataset
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# initialize the network
network = SampleNet(input_nodes=784, hidden_nodes=50, output_nodes=10)

# set hyperparameters
iters = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
iters_per_epoch = max(train_size / batch_size, 1)

# set lists for recording results
train_losses = []
train_accuracies = []
test_accuracies = []

print('accuracy to the training set | accuracy to the test set')

# echoing: repeat mini-batch gradient descenting
for i in range(iters):
    # create batches
    batch_mask = np.random.choice(train_size, batch_size) # Batch의 크기를 결정할 무작위 수 선택
    x_batch = x_train[batch_mask] # 학습용 배치 생성
    t_batch = t_train[batch_mask] # 답안 배치 생성

    # calculate gradients using forward & backprop
    grad = network.echo(x_batch, t_batch)

    # renew the parameters: weigths & biases
    for key in ('W1', 'b1', 'W2', 'b2'): # 문제!!!!
        network.params[key] -= learning_rate * grad[key] # 비용함수 상 다음 미분 지점을 지정

    loss = network.loss(x_batch, t_batch) # calcuate residuals(losses) between x_batch and t_batch
    train_losses.append(loss) # record losses

    # cross validation: get accuracies
    if i % iters_per_epoch == 0:
        '''
        현재 iters에서 채택된 예측 모델을 기준으로,
        현재 모델의 학습용 배치에 대한 예측 정확성과(train_acc)
        현재 모델의 답안 배치에 대한 예측 정확성(test_acc) 평가

        bias-variance tradeoff 원칙에 따라 train_acc, test_acc값이 비슷하게 나오는 모델이 최적의 성능 발휘
        overfitting: train_acc가 높고 test_acc가 낮을 경우 / low bias, high variance
        underfitting: train_acc가 낮고 test_acc가 높을 경우 / high bias, low variance
        '''
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    
        print(i, train_acc, test_acc)