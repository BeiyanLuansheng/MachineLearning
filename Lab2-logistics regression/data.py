import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_data(mean_neg, var_neg, size_neg, mean_pos, var_pos, size_pos, cov_xy=0.0):
    """
    生成数据
    """
    train_x = np.zeros((size_pos + size_neg, 2))
    train_y = np.zeros(size_pos + size_neg)
    train_x[:size_neg,:] = np.random.multivariate_normal(
        mean_neg, [[var_neg, cov_xy], [cov_xy, var_neg]], size=size_neg)
    train_x[size_neg:,:] = np.random.multivariate_normal(
        mean_pos, [[var_pos, cov_xy], [cov_xy, var_pos]], size=size_pos)
    train_y[size_neg:] = 1
    return train_x, train_y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss(train_x, train_y, w, lamda):
    """
    利用极大条件似然得到loss，并对loss做归一化
    """
    size = train_x.shape[0]
    W_dot_X = np.zeros((size, 1))
    ln_part = 0
    for i in range(size):
        W_dot_X[i] = w @ train_x[i].T
        ln_part += np.log(1 + np.exp(W_dot_X[i]))
    loss_mcle = train_y @ W_dot_X - ln_part
    return -loss_mcle / size

def gradient_descent(train_x, train_y, lamda, eta, epsilon, times=100000):
    """
    梯度下降
    :papam eta: 步长
    :param epsilon: 精度，算法终止距离
    :param times: 最大迭代次数
    :return 决策面方程系数数组和w
    """
    size = train_x.shape[0]
    dimension = train_x.shape[1]
    X = np.ones((size, dimension + 1)) # 构造X矩阵，第一维都设置成1，方便与w相乘
    X[:, 1:dimension+1] = train_x
    w = np.ones((1, X.shape[1]))
    new_loss = loss(X, train_y, w, lamda)
    for i in range(times):
        old_loss = new_loss
        t = np.zeros((size, 1))
        for j in range(size):
            t[j] = w @ X[j].T
        gradient_w = - (train_y - sigmoid(t.T)) @ X / size
        old_w = w
        w = w - eta * lamda * w - eta * gradient_w
        new_loss = loss(X, train_y, w, lamda)
        if old_loss < new_loss: #不下降了，说明步长过大
            w = old_w
            eta /= 2
            continue
        if old_loss - new_loss < epsilon:
            break
    print(i)
    w = w.reshape(dimension+1) # 得到的w是一个矩阵, 需要先改成行向量
    coefficient = -(w / w[dimension])[0:dimension] # 对w做归一化得到方程系数
    return coefficient, w

def accuracy(x, y, w):
    """
    计算准确率
    :param x: 测试数据
    :param y: 测试数据标签
    :param w: w
    """
    size = x.shape[0]
    dimension = x.shape[1]
    correct_count = 0
    X = np.ones((size, dimension + 1)) # 构造X矩阵，第一维都设置成1，方便与w相乘
    X[:, 1:dimension+1] = x
    for i in range(size):
        label = 0
        if w @ X[i].T >= 0:
            label = 1
        if label == y[i]:
            correct_count += 1
    accuracy = correct_count / size
    return accuracy

def show_2D(x, y, discriminant, title):
    """
    绘制二维图像
    """
    print('Discriminant function: y = ', discriminant)
    plt.scatter(x[:,0], x[:,1], c=y, s=30, marker='o', cmap=plt.cm.Spectral)
    real_x = min(x[:,0]) + (max(x[:,0]) - min(x[:,0])) * np.random.random(50)
    real_y = discriminant(real_x)
    plt.plot(real_x, real_y, 'r', label='discriminant')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=1)
    plt.title(title)
    plt.show()

def my_test(lamda, eta, epsilon, mean_neg, var_neg, size_neg, mean_pos, var_pos, size_pos, cov_xy=0.0):
    """
    利用高斯分布生成数据，进行测试
    """
    train_x, train_y = generate_data(mean_neg, var_neg, size_neg, mean_pos, var_pos, size_pos) # 生成数据
    coefficient, w = gradient_descent(train_x, train_y, lamda, eta, epsilon)
    discriminant = np.poly1d(coefficient[::-1])
    print('Train data accuracy:', accuracy(train_x, train_y, w))
    show_2D(train_x, train_y, discriminant, 'Train data')
    test_x, test_y = generate_data(mean_neg, var_neg, 2*size_neg, mean_pos, var_pos, 2*size_pos)
    print('Test data accuracy:', accuracy(test_x, test_y, w))
    show_2D(test_x, test_y, discriminant, 'Test data')

def uci_data(path):
    """
    从文件读取uci数据，分离训练数据集和测试数据集
    """
    data = np.loadtxt(path, dtype=np.int32)
    np.random.shuffle(data) # 随机打乱数据，便于选取数据
    dimension = data.shape[1]
    train_size = int(0.3 * data.shape[0]) # 按照3：7的比例分配训练集和测试集
    if train_size > 1000: # 数据量太大时，限制训练集的大小
        train_size = 1000
    train_data = data[:train_size, :]
    test_data = data[train_size:, :]
    train_x = train_data[:, 0:dimension-1]
    train_y = train_data[:, dimension-1] - 1
    test_x = test_data[:, 0:dimension-1]
    test_y = test_data[:, dimension-1] - 1
    return train_x, train_y, test_x, test_y

def show_3D(x, y, coefficient, title):
    """
    绘制三维图像
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x[:,0], x[:,1], x[:,2], c=y, cmap=plt.cm.Spectral)
    real_x = np.linspace(np.min(x[:,0])-20, np.max(x[:,0])+20, 255)
    real_y = np.linspace(np.min(x[:,1])-20, np.max(x[:,1])+20, 255)
    real_X, real_Y = np.meshgrid(real_x, real_y)
    real_z = coefficient[0] + coefficient[1] * real_X + coefficient[2] * real_Y
    ax.plot_surface(real_x, real_y, real_z, rstride=1, cstride=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()

def uci_test(path, lamda, eta, epsilon):
    """
    使用uci数据集测试
    """
    train_x, train_y, test_x, test_y = uci_data(path)
    coefficient, w = gradient_descent(train_x, train_y, lamda, eta, epsilon)
    print('Train data accuracy:', accuracy(train_x, train_y, w))
    show_3D(train_x, train_y, coefficient, 'Train data set')
    print('Test data accuracy:', accuracy(test_x, test_y, w))
    if test_x.shape[0] > 2000:
        show_3D(test_x[0:2000], test_y[0:2000], coefficient, 'Test data set (only show 2000 points)')
    else:
        show_3D(test_x, test_y, coefficient, 'Test data set')