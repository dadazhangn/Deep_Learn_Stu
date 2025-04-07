# def AND(X1, X2):
#     w1, w2, theta = 0.5, 0.5, 0.7
#     tmp = X1 * w1 + X2 * w2
#     if tmp <= theta:
#         return 0
#     elif tmp > theta:
#         return 1


# import numpy as np

# def AND(X1, X2):
#     x = np.array([X1, X2])
#     w = np.array([0.5, 0.5])
#     b = -0.7
#     tmp = np.sum(x * w) + b
#     if tmp <= 0:
#         return 0
#     else:
#         return 1
    
# def NAND(X1, X2):
#     x = np.array([X1, X2])
#     w = np.array([-0.5, -0.5])
#     b = 0.7
#     tmp = np.sum(x * w) + b
#     if tmp <= 0:
#         return 0
#     else:
#         return 1

# def OR(X1, X2):
#     x = np.array([X1, X2])
#     w = np.array([0.5, 0.5])
#     b = -0.2
#     tmp = np.sum(x * w) + b
#     if tmp <= 0:
#         return 0
#     else:
#         return 1

# def XOR(X1, X2):
#     s1 = NAND(X1, X2)
#     s2 = OR(X1, X2)
#     y = AND(s1, s2)
#     return y



# import numpy as np
# import matplotlib.pyplot as plt

# def step_function(x):
#     return np.array(x > 0, dtype=int)

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def relu(x):
#     return np.maximum(0, x)

# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)    
# y = sigmoid(x)  #sigmoid函数
# y = relu(x)  #relu函数
# # plt.plot(x, y)
# # plt.ylim(-0.1, 1.1)  #指定y轴范围
# # plt.show()

# A = np.array([[1, 2], [3, 4]])
# B = np.array([[5, 6], [7, 8]])  

# C = np.dot(A, B)  #矩阵乘法
# np.ndim(C)  # 2维
# print(C)  # [[19 22] [43 50]]









# if __name__ == "__main__":
    # ret = AND(0, 0)
    # print(ret)  # 0

# import numpy as np

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def  identity_function(x):
#     return x


# def init_network():
#     network = {}
#     network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
#     network['b1'] = np.array([0.1, 0.2, 0.3])
#     network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
#     network['b2'] = np.array([0.1, 0.2])
#     network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
#     network['b3'] = np.array([0.1, 0.2])
#     return network

# def forward(network, x):
#     W1, W2, W3 = network['W1'], network['W2'], network['W3']
#     b1, b2, b3 = network['b1'], network['b2'], network['b3']

#     a1 = np.dot(x, W1) + b1
#     z1 = sigmoid(a1)
#     a2 = np.dot(z1, W2) + b2
#     z2 = sigmoid(a2)
#     a3 = np.dot(z2, W3) + b3

#     y = identity_function(a3)  # 输出层使用恒等函数
    
#     return y
    
# network = init_network()
# x = np.array([1.0, 0.5])
# y = forward(network, x)
# print(y)  # 输出结果


# def softmax(a):
#     c = np.max(a)  # 为了数值稳定性，减去最大值
#     exp_a = np.exp(a - c)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#     return y

# import sys, os
# sys.path.append(os.pardir)
# from dataset.mnist import load_mnist
# import numpy as np
# from PIL import Image


# def img_show(img):
#     pil_img = Image.fromarray(np.uint8(img))  # 将numpy数组转换为PIL图像对象
#     pil_img.show()  # 显示图像

# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True, one_hot_label=False)
# # print(x_train.shape)  # (60000, 784)
# # print(t_train.shape)  # (60000, 10)
# # print(x_test.shape)   # (10000, 784)
# # print(t_test.shape)   # (10000, 10)
# img = x_train[0]
# label = t_train[0]
# print(label)
# print(img.shape)  # (784,)
# img = img.reshape(28, 28)  # 将784维数组转换为28*28的二维数组
# print(img.shape)  # (28, 28)
# img_show(img)

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True, one_hot_label=True)

print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000, 10)

train_size = x_train.shape[0]  # 训练集大小
# print(train_size)  # 60000


