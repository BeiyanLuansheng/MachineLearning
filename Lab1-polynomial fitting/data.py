import numpy as np
import math
import matplotlib.pyplot as plt

def generate_data(order, size, mu=0, sigma=0.2, begin=0, end=1):
    """
    生成数据
    :param order: 多项式阶数
    :param size: 训练集大小
    :param mu: 高斯分布均值
    :param sigma: 高斯分布标准差
    :param begin: 区间起点
    :param end: 区间终点
    :return: x, y, x = (size, order+1), y = (size, 1)
    """
    x = np.linspace(begin, end, size)
    guass_noise = np.random.normal(mu, sigma, size) # 高斯分布噪声
    y = np.sin(2 * np.pi * x) + guass_noise
    train_y = y.reshape(size, 1)
    train_x = np.zeros((size, order + 1))
    nature_row = np.arange(0, order+1)
    for i in range(size):
        row = np.ones(order + 1) * x[i]
        row = row ** nature_row
        train_x[i] = row
    return train_x, train_y, x, y

def cal_w(train_x, train_y, lamda):
    """
    计算w
    """
    return np.linalg.inv(np.dot(train_x.T, train_x) + lamda * np.eye(train_x.shape[1])).dot(train_x.T).dot(train_y)

def lsm(train_x, train_y, lamda=0):
    """
    最小二乘法求解析解
    lamda=0时不带正则项，lamda!=0时带正则项，令损失函数导数等于0，求w的最优解
    """
    w = cal_w(train_x, train_y, lamda)
    fitting_polynomial = np.poly1d(w[::-1].reshape(train_x.shape[1]))
    return fitting_polynomial

def loss(train_x, train_y, w, lamda=0):
    """
    求损失函数 E(w) = 1/2 * (Xw - Y)^T * (Xw - Y)  + lambda / 2 (w^T * w)
    :return loss
    """
    loss = train_x.dot(w) - train_y
    loss = 1/2 * np.dot(loss.T, loss) + lamda / 2 * np.dot(w.T, w)
    return loss

def gradient_descent(train_x, train_y, lamda, alpha, epsilon, times=10000000):
    """
    梯度下降
    :papam alapha: 步长
    :param epsilon: 精度，算法终止距离
    :param times: 最大迭代次数
    """
    w = np.zeros((train_x.shape[1], 1))
    new_loss = abs(loss(train_x, train_y, w, lamda))
    for i in range(times):
        old_loss = new_loss
        gradient_w = np.dot(train_x.T, train_x).dot(w) - np.dot(train_x.T, train_y) + lamda * w
        old_w = w
        w -= gradient_w * alpha
        new_loss = abs(loss(train_x, train_y, w, lamda))
        if old_loss < new_loss: #不下降了，说明步长过大
            w = old_w
            alpha /= 2
        if old_loss - new_loss < epsilon:
            break
    print(i)
    fitting_polynomial = np.poly1d(w[::-1].reshape(train_x.shape[1]))
    return fitting_polynomial

def conjugate_gradient(train_x, train_y, lamda, epsilon):
    """
    共轭梯度
    :param epsilon: 精度
    """
    # 记为Aw=b的形式，其中A = X^T * X + lambda，b = X^T * Y
    A = np.dot(train_x.T, train_x) + lamda * np.eye(train_x.shape[1]) # n+1 * n+1
    b = np.dot(train_x.T, train_y) # n+1 * 1
    w = np.zeros((train_x.shape[1], 1)) #  初始化w为 n+1 * 1 的零阵
    r = b
    p = b
    i=0
    while True:
        i = i +1
        norm_2 = np.dot(r.T, r)
        a = norm_2 / np.dot(p.T, A).dot(p)
        w = w + a * p
        r = r - (a * A).dot(p)
        if r.T.dot(r) < epsilon:
            break
        b = np.dot(r.T, r) / norm_2
        p = r + b * p
    print(i)
    fitting_polynomial = np.poly1d(w[::-1].reshape(train_x.shape[1]))
    return fitting_polynomial

def plt_show(x, y, poly_fit, title):
    """
    多项式绘图
    """
    plot1 = plt.plot(x, y, 'co', label='train_data')
    real_x = np.linspace(0, 1, 50)
    real_y = np.sin(real_x * 2 * np.pi)
    fit_y = poly_fit(real_x)
    plot2 = plt.plot(real_x, fit_y, 'b', label='fit_result')
    plot3 = plt.plot(real_x, real_y, 'r', label='real_data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=1)
    plt.title(title)
    plt.show()

def rms(train_x, train_y, w, lamda, size):
    """
    求均方根E_rms
    """
    Ew = loss(train_x, train_y, w)
    rms = np.sqrt(2 *  Ew / size)
    return rms

def rms_show(train_x, train_y, validation_x, validation_y, train_size, validation_size):
    """
    绘制E_rms
    """
    ln_lamda = np.linspace(-40, 0, 41)
    rms_train = np.zeros(41)
    rms_validation = np.zeros(41)
    for i in range(0, 41):
        lamda = np.exp(ln_lamda[i])
        w = cal_w(train_x, train_y, lamda)
        rms_train[i] = rms(train_x, train_y, w, lamda, train_size)
        rms_validation[i] = rms(validation_x, validation_y, w, lamda, validation_size)
    train_plot = plt.plot(ln_lamda, rms_train, 'b', label='train')
    validation_plot = plt.plot(ln_lamda, rms_validation, 'r', label = 'validation')
    min_rms = np.min(rms_validation)
    min_rms_index = np.where(rms_validation == min_rms)
    exp = min_rms_index[0][0] - 41
    title = '$Min: e^{' + str(exp) + '}, ' + '{:.3f}'.format(min_rms) +'$'
    plt.xlabel('$\ln \lambda$')
    plt.ylabel('$E_{rms}$')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

def method_compare_show(x, y, lsm_fit, lsm_punish, gd_fit, cg_fit, title):
    """
    对比四种拟合方法效果，绘制图像
    """
    plot0 = plt.plot(x, y, 'co')
    real_x = np.linspace(0, 1, 50)
    real_y = np.sin(real_x * 2 * np.pi)
    plot1 = plt.plot(real_x, real_y, 'r--', label='real_data')
    lsm_fit_y = lsm_fit(real_x)
    lsm_punish_y = lsm_punish(real_x)
    gd_fit_y = gd_fit(real_x)
    cg_fit_y = cg_fit(real_x)
    plot2 = plt.plot(real_x, lsm_fit_y, label='lsm_fit')
    ploy3 = plt.plot(real_x, lsm_punish_y, label='lsm_punish')
    plot4 = plt.plot(real_x, gd_fit_y, '-.', label='gradient_descent')
    plot5 = plt.plot(real_x, cg_fit_y, label='conjugate_gradient')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=1)
    plt.title(title)
    plt.show()