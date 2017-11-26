# fisher using kernel method
# by zsz
# 2017-11-14

from numpy import *
from collections import Iterable
import matplotlib.pyplot as plt

# fetch training data
def fetchData(filename, xBeg, xSize, yBeg, ySize):
    raw = loadtxt(filename)
    x_data = raw[:, xBeg:xBeg+xSize]
    y_data = raw[:, yBeg:yBeg+ySize]
    return mat(x_data), mat(y_data)

# conventional class
class Fisher:
    def __init__(self, name = 'fisher'):
        self.name = name

    # use this function to fit to data
    def fit(self, x1_data, x2_data, kernel = None, t = 1):
        # determine the kernel and linear is default
        if kernel == None:
            self.kernel_name = 'linear'
        elif not isinstance(kernel, Iterable):
            raise NameError("expected kernel parameter is ['kernel_name', other params...]")
        elif kernel[0] == 'linear':
            self.kernel_name = 'linear'
        elif kernel[0] == 'rbf':
            self.kernel_name = 'rbf'
            if len(kernel) < 2:
                raise NameError("give rbf kernel without the value of sigma")
            if float(kernel[1]) <= 0:
                self.sigma = 1.0
            else:
                self.sigma = float(kernel[1])
        elif kernel[0] == 'polynomial':
            self.kernel_name = 'polynomial'
            if len(kernel) < 2:
                raise NameError("give polynomial kernel without index m")
            self.m = int(kernel[1])
        else:
            raise NameError("no support for kernel : {}".format(kernel[0]))

        # get two types of data
        self.x1_data = mat(x1_data.copy())
        self.N1 = len(x1_data)
        self.x2_data = mat(x2_data.copy())
        self.N2 = len(x2_data)
        self.N = self.N1 + self.N2
        self.x_data = row_stack([self.x1_data, self.x2_data])
        self.t = t

        # solve the function f(x) = w * x + b
        self.calculate_w()
        self.calculate_b()


    # use this function to predict
    # use the function of f(x) = w * x + b
    def predict(self, x):
        x = mat(x)
        tmp = mat(zeros((self.N, 1)))
        for i in range(self.N):
            tmp[i, 0] = self.K(self.x_data[i, :], x)
        return dot(self.w, tmp) + self.b


    # to solve K(x, z) which is non-linear function
    def K(self, x, z):
        x = mat(x)
        z = mat(z)
        if self.kernel_name == 'linear':
            return dot(x, z.T)
        if self.kernel_name == 'rbf':
            diff = x - z
            return exp(-dot(diff, diff.T) / (2 * self.sigma**2))
        if self.kernel_name == 'polynomial':
            inner = dot(x, z.T)
            return (inner + 1)**self.m


    # to solve the matrix of Gamma
    def calculateGamma(self):
        self.Gamma = zeros((self.N, 1))
        for i in range(self.N):
            a = 0
            for j in range(self.N1):
                a = a + self.K(self.x_data[i, :], self.x1_data[j, :])
            a = a / self.N1
            b = 0
            for j in range(self.N2):
                b = b + self.K(self.x_data[i, :], self.x2_data[j, :])
            b = b / self.N2
            self.Gamma[i, 0] = a - b


    # to solve the matrix of Ns
    def calculateNmat(self):
        N1 = mat(zeros((self.N, self.N)))
        N2 = mat(zeros((self.N, self.N)))
        for i in range(self.N1):
            sum = mat(zeros((1, self.N)))
            for j in range(self.N):
                a = self.K(self.x_data[j, :], self.x1_data[i, :])
                b = 0
                for k in range(self.N1):
                    b = b + self.K(self.x_data[j, :], self.x1_data[k, :])
                sum[0, j] = a - b/self.N1
            N1 = N1 + dot(sum.T, sum)

        for i in range(self.N2):
            sum = mat(zeros((1, self.N)))
            for j in range(self.N):
                a = self.K(self.x_data[j, :], self.x2_data[i, :])
                b = 0
                for k in range(self.N2):
                    b = b + self.K(self.x_data[j, :], self.x2_data[k, :])
                sum[0, j] = a - b/self.N2
            N2 = N2 + dot(sum.T, sum)
        self.N1mat = N1
        self.N2mat = N2
        self.Nmat = N1 + N2


    # to solve the matrix of K
    def calculateKmat(self):
        Kmat = mat(zeros((self.N, self.N)))
        for i in range(self.N):
            for j in range(self.N):
                Kmat[i, j] = self.K(self.x_data[i, :], self.x_data[j, :])
        self.Kmat = Kmat


    # to solve the tensor of w
    def calculate_w(self):
        self.calculateGamma()
        self.calculateKmat()
        self.calculateNmat()
        # print('Nmat is {}'.format(self.Nmat))
        # print('Kmat is {}'.format(self.Kmat))
        # print('t is {}'.format(self.t))
        self.w = dot(self.Gamma.T, linalg.inv((self.Nmat + self.t * self.Kmat).T))


    # to solve bias
    def calculate_b(self):
        tmp = mat(sum(self.Kmat, 1))
        self.b = -dot(self.w, tmp) / self.N
        # print('b = {}'.format(self.b))


def main():
    # here we use the 2-D data
    x_data, y_data = fetchData('sample.txt', 0, 3, 3, 1)
    x1_data = x_data[:50, :]
    x2_data = x_data[50:, :]
    fisher = Fisher()
    # train the fisher
    fisher.fit(x1_data, x2_data, ['rbf', 1])
    # calculate the empirical risk
    errorCount = 0
    for i in range(len(x_data)):
        res = fisher.predict(x_data[i, :])
        if (y_data[i, :] == 0 and res < 0) or (y_data[i, :] == 1 and res > 0):
            errorCount = errorCount + 1
    print('the empirical risk is {}'.format(float(errorCount) / len(x_data)))

    # x = linspace(-10, 10, 100)
    # y = linspace(-10, 10, 100)
    # edge = []
    # find the edge
    # for i in range(len(x)):
    #    for j in range(len(y)):
    #        if abs(fisher.predict([x[i], y[j]])) < 0.005:
    #            edge.append([x[i], y[j]])
    # edge = mat(edge)
    # show it up
    # plt.scatter(list(x1_data[:, 0]), list(x1_data[:, 1]), c='r')
    # plt.scatter(list(x2_data[:, 0]), list(x2_data[:, 1]), c='g')
    # plt.scatter(list(edge[:, 0]), list(edge[:, 1]), c='b')
    # plt.show()



if __name__ == '__main__':
    main()