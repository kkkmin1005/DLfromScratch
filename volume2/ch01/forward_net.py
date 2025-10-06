import numpy as np
#설계 규칙 - 모든 계층은 forward 함수를 가짐, 객체 생성시 파라미터를 객체 변수로 가짐

#sigmoid class - 순전파로 시그모이드 계산만, 별도의 파라미터 저장x
class Sigmoid():
    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1/(1 + np.exp(-x))
    
#fc layer - 순전파로 x @ W + b, params에 파라미터 저장
class Affine():
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.matmul(x,W) + b
        return out

class TwoLayerNet():
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        #가중치 초기화
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        #계층생성
        self.layes = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        #파라미터 저장
        self.params = []
        for layer in self.layers:
            self.params += layer.params
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x