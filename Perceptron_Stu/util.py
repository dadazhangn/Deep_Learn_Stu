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
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]  # 批量大小
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size  # 1e-7是为了避免log(0)的情况


def numerical_diff(f, x):
    h = 1e-4  # 微小的值
    return (f(x + h) - f(x - h)) / (2 * h)


# def numerical_gradient(f, x):
#     h = 1e-4  # 微小的值
#     grad = np.zeros_like(x)  # 创建与x形状相同的零数组
#     for idx in range(x.size):
#         tmp_val = x[idx]  # 保存原始值
#         x[idx] = tmp_val + h  # 增加h
#         fxh1 = f(x)  # f(x+h)
#         x[idx] = tmp_val - h  # 减去h
#         fxh2 = f(x)  # f(x-h)
#         grad[idx] = (fxh1 - fxh2) / (2 * h)  # 中心差分法计算梯度
#         x[idx] = tmp_val  # 恢复原始值

#     return grad

import numpy as np

def numerical_gradient(f, x):
    h = 1e-4  # 微小的值
    grad = np.zeros_like(x)  # 创建与x形状相同的零数组

    # 遍历x的每个维度
    it = np.nditer(x, flags=['multi_index'])  # 使用nditer遍历多维数组
    while not it.finished:
        idx = it.multi_index  # 获取当前的索引
        tmp_val = x[idx]  # 保存原始值
        x[idx] = tmp_val + h  # 增加h
        fxh1 = f(x)  # f(x+h)
        x[idx] = tmp_val - h  # 减去h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)  # 中心差分法计算梯度
        x[idx] = tmp_val  # 恢复原始值
        it.iternext()  # 移动到下一个元素

    return grad



def gradient_descent(f, x, lr=0.01, step_num=100):
    for i in range(step_num):
        x -= lr * numerical_gradient(f, x)  # 更新x
    return x