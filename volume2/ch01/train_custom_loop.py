import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from common.optimizers import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet

#하이퍼파라미터 세팅
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

#데이터로드, 모델과 옵티마이저 생성
x, t= spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=lr)

#학습에 사용되는 변수
data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    #데이터 섞기
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iter in range(max_iters):
        batch_x = x[iter*batch_size:(iter+1)*batch_size]
        batch_t = t[iter*batch_size:(iter+1)*batch_size]

        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        if (iter+1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print('| epoch %d | iters %d / %d | loss %.2f' % (epoch+1, iter+1, max_iters, avg_loss))
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0