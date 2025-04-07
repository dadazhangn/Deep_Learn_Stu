import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
import pickle


def get_data():
    # 读取数据集
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=False)
    
    return x_test, t_test

def init_work():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)  # 为了数值稳定性，减去最大值
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3

    y = softmax(a3)  # 输出层使用softmax函数
    
    return y

x, y= get_data()
network = init_work()

batch_size = 100  # 批量大小
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i:i + batch_size]
    y_batch = predict(network, x_batch)  # 预测结果
    p = np.argmax(y_batch, axis=1)  # 预测结果
    accuracy_cnt +=  np.sum(p == y[i:i + batch_size])  # 计算准确率
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # 准确率
