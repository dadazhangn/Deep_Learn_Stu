import numpy as np

def step_function(x):
    return np.array(x > 0, dtype=int)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)  # 为了数值稳定性，减去最大值
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def  cross_entropy_error(y, t):
    delta = 1e-7  # 防止log(0)的情况
    return -np.sum(t * np.log(y + delta))  # 交叉熵损失函数