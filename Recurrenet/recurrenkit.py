# recurrentkit :: recurrent neural layers module #
# Based on Numpy #
# Austin Hyeon, 2020. "Sociology meets Computer Science" #

import numpy as np
from funckit import softmax
from cellkit import *

class VanillaRecurrent:

    ''' recurrent layer made up of the simple cells '''

    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wh), np.zeros_like(Wx), np.zeros_like(b)]
        self.cells = None

        self.h, self.dh = None, None
        self.stateful = stateful


    def set_state(self, h):
        self.h = h


    def reset_state(self):
        self.h = None

    
    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        # N: size of mini-batch
        # T: length of the input time series data
        # D: the number of dimensions
        D, H = Wx.shape

        self.cells = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        
        for t in range(T):
            cell = VanillaCell(*self.params)
            self.h = cell.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.cells.append(cell)

        return hs


    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]

        for t in reversed(range(T)):
            cell = self.cells[t]
            dx, dh = cell.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(cell.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        
        self.dh = dh

        return dxs


class LSTMRecurrent:

    ''' recurrent layer made up of the LSTMs '''

    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful


    def set_state(self, h, c=None):
        self.h, self.c = h, c


    def reset_state(self):
        self.h, self.c = None, None


    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTMCell(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)

        return hs


    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs


class GRURecurrent:
    
    ''' recurrent layer made up of the GRUs '''

    def __init__(self):
        None


class AffineRecurrent:

    ''' affine recurrent layer '''

    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None


    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N*T, -1)
        out = np.matmul(rx, W) + b
        self.x = x
        
        return out.reshape(N, T, -1)


    def backward(self, dstream):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dstream = dstream.reshape(N*T, -1)
        rx = x.reshape(N*T, -1)

        db = np.sum(dstream, axis=0)
        dW = np.matmul(rx.T, dstream)
        dx = np.matmul(dstream, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class EmbedRecurrent:

    ''' embedding matrix recurrent layer '''

    def __init__(self, EMW):
        self.params = [EMW] # EMW: embedding matrix weights
        self.grads = [np.zeros_like(EMW)]
        self.layers = None
        self.EMW = EMW

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.EMW.shape

        out = np.empty((N, T, D), dtype='f')
        self.cells = []

        for t in range(T):
            cell = EmbeddingCell(self.EMW)
            out[:, t, :] = cell.forward(xs[:, t])
            self.cells.append(cell)

        return out

    def backward(self, dstream):
        N, T, D = dstream.shape

        grad = 0
        for t in range(T):
            cell = self.cells[t]
            cell.backward(dstream[:, t, :])
            grad += cell.grads[0]

        self.grads[0][...] = grad

        return None


class SoftmaxLossRecurrent:

    ''' softmax-with-loss recurrent layer '''

    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 정답 레이블이 원핫 벡터인 경우
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # 배치용과 시계열용을 정리(reshape)
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_label에 해당하는 데이터는 손실을 0으로 설정
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))

        return loss

    def backward(self, dstream=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dstream
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_label에 해당하는 데이터는 기울기를 0으로 설정

        dx = dx.reshape((N, T, V))

        return dx


class DropoutRecurrent:

    ''' dropout recurrent layer '''

    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True

    def forward(self, xs):
        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale

            return xs * self.mask
        else:
            return xs

    def backward(self, dstream):
        return dstream * self.mask