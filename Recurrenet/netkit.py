import sys, os
sys.path.append('..')

import numpy as np
import pickle
from recurrenkit import *

class VanillaRNN:

    ''' RNN with simple recurrent cells '''

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        ''' set parameters '''

        EMW = (rn(V, D) / 100).astype('f') # init the embedding matrix elements(= weights)
        vanilla_Wx = (rn(D, H) / np.sqrt(D)).astype('f') # init weights for input vectors(= the vectors of embedding matrix)
        vanilla_Wh = (rn(H, H) / np.sqrt(H)).astype('f') # init weights for hidden states
        vanilla_b = np.zeros(H).astype('f') # init biases for input vectors
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f') # init weigths for affine operation
        affine_b = np.zeros(V).astype('f') # init biases for affine operation

        ''' set recurrent layers '''

        self.layers = [
            EmbedRecurrent(EMW),
            VanillaRecurrent(vanilla_Wx, vanilla_Wh, vanilla_b, stateful=True),
            AffineRecurrent(affine_W, affine_b)
        ]
        self.loss_layer = SoftmaxLossRecurrent()
        self.vanilla_layer = self.layers[1] # VanillaRecurrent()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params # merge all parameters of each layer
            self.grads += layer.grads # merge all gradients of each layer

    
    def reset_state(self):
        self.vanilla_layer.reset_state()

    
    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)

        # the FOR statement eqauls to the bellow

        # W[idx] = EmbedRecurrent.forward(xs) :: return the index of a vector on the weights matrix "W[idx]"
        # hs = VanillaRecurrent.forward(W[idx]) :: retrun the hidden state "hs"
        # xs = AffineRecurrent.forward(hs) :: retrun affine operation(XW + b) of W[0] to W[n]
        # loss = SoftmaxLossRecurrent.forward(xs, ts) :: loss between the predicted data(xs) and the actual data(ts)

        return loss


    def backward(self, dstream=1):
        dstream = self.loss_layer.backward(dstream)

        for layer in reversed(self.layers):
            dstream = layer.backward(dstream)

        # the FOR statement eqauls to the bellow

        # dLoss = SoftmaxLossRecurrent.backward(dstream=1) # return derivative of total loss
        # dx = AffineRecurrent.backward(dLoss) # return derivatives of affine operations
        # dxs = VanillaRecurrent.backward(dx) # return derivatives of the hidden state & the input vector
        # dEMW = EmbedRecurrent.backward(dxs) # return derivatives of all embedding matrix weights
        
        return dstream


class LSTMRecurrentNet:

    ''' RNN with LSTM cells '''

    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        ''' set parameters '''
        lstm_Wx, lstm_Wh, lstm_b = [], [], []

        EMW = (rn(V, D) / 100).astype('f') # init the embedding matrix elements(= weights)
        for i in range(2):
            lstm_Wx.append((rn(D, 4 * H) / np.sqrt(D)).astype('f')) # init weights for input vectors(= the vectors of embedding matrix)
            lstm_Wh.append((rn(H, 4 * H) / np.sqrt(H)).astype('f')) # init weights for hidden states
            lstm_b.append(np.zeros(4 * H).astype('f')) # init biases for input vectors
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f') # init weigths for affine operation
        affine_b = np.zeros(V).astype('f') # init biases for affine operation

        ''' set recurrent layers '''

        self.layers = [
            EmbedRecurrent(EMW),
            DropoutRecurrent(dropout_ratio=0.5),
            LSTMRecurrent(lstm_Wx[0], lstm_Wh[0], lstm_b[0], stateful=True),
            DropoutRecurrent(dropout_ratio=0.5),
            LSTMRecurrent(lstm_Wx[1], lstm_Wh[1], lstm_b[1], stateful=True),
            DropoutRecurrent(dropout_ratio=0.5),
            AffineRecurrent(affine_W, affine_b)
        ]
        self.loss_layer = SoftmaxLossRecurrent()
        self.lstm_layers = [self.layers[2], self.layers[4]] # LSTMRecurrent()
        self.dropout_layers = [self.layers[1], self.layers[3], self.layers[5]] # DropoutRecurrent()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params # merge all parameters of each layer
            self.grads += layer.grads # merge all gradients of each layer

    
    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()


    def predict(self, xs, train_flg=False):
        for layer in self.dropout_layers:
            layer.train_flg = train_flg

        for layer in self.layers:
            xs = layer.forward(xs)

        return xs


    def forward(self, xs, ts, train_flg=True):
        score = self.predict(xs, train_flg)
        loss = self.loss_layer.forward(score, ts)

        return loss


    def backward(self, dstream=1):
        dstream = self.loss_layer.backward(dstream)
        for layer in reversed(self.layers):
            dstream = layer.backward(dstream)

        return dstream
        

    def pkl_save_params(self, pickle_file='rnn_records.pkl'):
        with open(pickle_file, 'rb') as f:
            pickle.dump(self.params, f)


    def pkl_load_params(self, pickle_file='rnn_records.pkl'):
        with open(pickle_file, 'rb') as f:
            self.params = pickle.load(f)
