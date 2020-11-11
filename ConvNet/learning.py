# learning!

import sys, os
sys.path.append(os.pardir)
sys.path.insert(0, '/Users/austin/OneDrive/Repositories/MNIST/') # for macOS
sys.path.insert(0, '/Users/User/Documents/GitHub/MNIST') # for Windows

import numpy as np
from netkit import ConvNet
from optimakit import *
from trainkit import Trainer

from mnist import load_mnist


# load the dataset
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 5

network = ConvNet(input_dim=(1,28,28), 
                        conv_params={'filter_num': 30, 'filter_size': 5, 'padding': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'learning_rate': 0.001},
                  evaluate_sample_num_per_epoch=1000)

trainer.run()