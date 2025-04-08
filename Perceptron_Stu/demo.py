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

# import sys, os
# sys.path.append(os.pardir)
# from dataset.mnist import load_mnist

# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True, one_hot_label=True)

# print(x_train.shape)  # (60000, 784)
# print(t_train.shape)  # (60000, 10)

# train_size = x_train.shape[0]  # 训练集大小
# # print(train_size)  # 60000
# batch_size = 10  # 批量大小
# batch_mask = np.random.choice(train_size, batch_size)  # 随机选择10个样本
# x_train_batch = x_train[batch_mask]  # 选取对应的训练数据
# t_train_batch = t_train[batch_mask]  # 选取对应的标签数据

# import numpy as np
# from util import *

# class simpleNet:
#     def __init__(self):
#         self.W = np.random.rand(2, 3)

#     def predict(self, x):
#         y = np.dot(x, self.W)
#         return y
    
#     def loss(self, x, t):
#         z = self.predict(x)
#         y = softmax(z)
#         loss = cross_entropy_error(y, t)
#         return loss


# if __name__ == "__main__":
#     net = simpleNet()
#     print(net.W)  # 随机初始化权重
#     x = np.array([0.6, 0.9])
#     p = net.predict(x)  # 预测结果
#     print("p: ",p)  # 预测结果
#     print("p max index: ",np.argmax(p))  # 预测结果最大值的索引
#     t = np.array([0, 0, 1])  # 真实标签
#     print("loss: ",net.loss(x, t))  # 损失值

#     def f(W):
#         return net.loss(x, t)

#     dw = numerical_gradient(f, net.W)  # 数值梯度
#     print("dw: ", dw)  # 数值梯度


# import sys, os
# sys.path.append(os.pardir)
# from util import *
# import numpy as np

# class TwoLayerNet:
#     def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
#         self.params = {}
#         self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) 
#         self.params['b1'] = np.zeros(hidden_size)
#         self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
#         self.params['b2'] = np.zeros(output_size)
    
#     def predict(self, x):
#         W1, W2 = self.params['W1'], self.params['W2']
#         b1, b2 = self.params['b1'], self.params['b2']

#         a1 = np.dot(x, W1) + b1
#         z1 = sigmoid(a1)
#         a2 = np.dot(z1, W2) + b2
#         y = softmax(a2)
        
#         return y
    
#     def loss(self, x, t):
#         y = self.predict(x)
#         return cross_entropy_error(y, t)
    
#     def accuracy(self, x, t):
#         y = self.predict(x)
#         y = np.argmax(y, axis=1)
#         t = np.argmax(t, axis=1)
#         accuracy = np.sum(y == t) / float(x.shape[0])
#         return accuracy
    
#     def numerical_gradient(self, x, t):
#         loss_W = lambda W: self.loss(x, t)
#         grads = {}
#         grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
#         grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
#         grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
#         grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

#         return grads
    

# if __name__ == "__main__":
#     net  = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
#     net.params['W1'].shape  # (784, 100)
#     net.params['b1'].shape  # (100,)
#     net.params['W2'].shape  # (100, 10)
#     net.params['b2'].shape  # (10,)

#     x = np.random.randn(100, 784)  # 100个样本
#     y = net.predict(x)  # 预测结果


from layer_naive import MulLayer, AddLayer
apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()

mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple, apple_num)  # 苹果的总价
price = mul_tax_layer.forward(apple_price, tax)  # 苹果的总价加上税

print(price)  # 220.0


dprice = 1





