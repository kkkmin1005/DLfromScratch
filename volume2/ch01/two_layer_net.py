import sys
sys.path.append('..')
import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        #가중치 초기화
        self.W1 = np.random.randn(I, H)
        self.b1 = np.random.randn(H)
        self.W2 = np.random.randn(H, O)
        self.b2 = np.random.randn(O)

        #계층 생성
        self.layers = [
            Affine(self.W1, self.b1),
            Sigmoid(),
            Affine(self.W2, self.b2)
        ]
        self.loss_layer = SoftmaxWithLoss()

        #파라미터 저장
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss
    
    def backward(self, dout = 1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout