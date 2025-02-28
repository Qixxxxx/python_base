import matplotlib.pyplot as plt
import numpy as np


class ActivateFunc():
    def __init__(self, x, b=None, lamb=None, alpha=None, a=None):
        super(ActivateFunc, self).__init__()
        self.x = x
        self.b = b
        self.lamb = lamb
        self.alpha = alpha
        self.a = a

    def Sigmoid(self):
        y = np.exp(self.x) / (np.exp(self.x) + 1)
        y_grad = y * (1 - y)
        return [y, y_grad]

    def Tanh(self):
        y = np.tanh(self.x)
        y_grad = 1 - y * y
        return [y, y_grad]

    def Swish(self):  # b是一个常数，指定b
        y = self.x * (np.exp(self.b * self.x) / (np.exp(self.b * self.x) + 1))
        y_grad = np.exp(self.b * self.x) / (1 + np.exp(self.b * self.x)) + self.x * (
                self.b * np.exp(self.b * self.x) / ((1 + np.exp(self.b * self.x)) * (1 + np.exp(self.b * self.x))))
        return [y, y_grad]

    def ELU(self):  # alpha是个常数，指定alpha
        y = np.where(self.x > 0, self.x, self.alpha * (np.exp(self.x) - 1))
        y_grad = np.where(self.x > 0, 1, self.alpha * np.exp(self.x))
        return [y, y_grad]

    def SELU(self):  # lamb大于1，指定lamb和alpha
        y = np.where(self.x > 0, self.lamb * self.x, self.lamb * self.alpha * (np.exp(self.x) - 1))
        y_grad = np.where(self.x > 0, self.lamb * 1, self.lamb * self.alpha * np.exp(self.x))
        return [y, y_grad]

    def ReLU(self):
        y = np.where(self.x < 0, 0, self.x)
        y_grad = np.where(self.x < 0, 0, 1)
        return [y, y_grad]

    def PReLU(self):  # a大于1，指定a
        y = np.where(self.x < 0, self.x / self.a, self.x)
        y_grad = np.where(self.x < 0, 1 / self.a, 1)
        return [y, y_grad]

    def LeakyReLU(self):  # a大于1，指定a
        y = np.where(self.x < 0, self.x / self.a, self.x)
        y_grad = np.where(self.x < 0, 1 / self.a, 1)
        return [y, y_grad]

    def Mish(self):
        f = 1 + np.exp(self.x)
        y = self.x * ((f * f - 1) / (f * f + 1))
        y_grad = (f * f - 1) / (f * f + 1) + self.x * (4 * f * (f - 1)) / ((f * f + 1) * (f * f + 1))
        return [y, y_grad]

    def ReLU6(self):
        y = np.where(np.where(self.x < 0, 0, self.x) > 6, 6, np.where(self.x < 0, 0, self.x))
        y_grad = np.where(self.x > 6, 0, np.where(self.x < 0, 0, 1))
        return [y, y_grad]

    def Hard_Swish(self):
        f = self.x + 3
        relu6 = np.where(np.where(f < 0, 0, f) > 6, 6, np.where(f < 0, 0, f))
        relu6_grad = np.where(f > 6, 0, np.where(f < 0, 0, 1))
        y = self.x * relu6 / 6
        y_grad = relu6 / 6 + self.x * relu6_grad / 6
        return [y, y_grad]

    def Hard_Sigmoid(self):
        f = (2 * self.x + 5) / 10
        y = np.where(np.where(f > 1, 1, f) < 0, 0, np.where(f > 1, 1, f))
        y_grad = np.where(f > 0, np.where(f >= 1, 0, 1 / 5), 0)
        return [y, y_grad]

    def GELU(self):
        y = 0.5 * self.x * (1 + np.tanh(np.sqrt(2 / np.pi) * (self.x + 0.044715 * self.x ** 3)))
        y_grad = (
                     (np.tanh((np.sqrt(2) * (0.044715 * self.x ** 3 + self.x)) / np.sqrt(np.pi)) + (
                                 (np.sqrt(2) * self.x * (
                                         0.134145 * self.x ** 2 + 1) * ((1 / np.cosh(
                                     (np.sqrt(2) * (0.044715 * self.x ** 3 + self.x)) / np.sqrt(
                                         np.pi))) ** 2)) / np.sqrt(
                             np.pi) + 1))) / 2
        return [y, y_grad]


def PlotActiFunc(x, y, title):
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    plt.plot(x, y)
    plt.title(title)
    plt.show()


def PlotMultiFunc(x, y1, y2, title):
    plt.figure()
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    plt.plot(x, y1)
    plt.plot(x, y2, color='red')
    plt.title(title)
    plt.legend([title, title + '-grad'])
    plt.show()


if __name__ == '__main__':
    x = np.arange(-10, 10, 0.01)
    activateFunc = ActivateFunc(x)
    activateFunc.b = 1
    activateFunc.alpha = 1.67326
    activateFunc.lamb = 1.0507
    activateFunc.a = 100

    PlotMultiFunc(x, activateFunc.Sigmoid()[0], activateFunc.Sigmoid()[1], "Sigmoid")
    PlotMultiFunc(x, activateFunc.Tanh()[0], activateFunc.Tanh()[1], "Tanh")
    PlotMultiFunc(x, activateFunc.Hard_Sigmoid()[0], activateFunc.Hard_Sigmoid()[1], "Hard_Sigmoid")
    PlotMultiFunc(x, activateFunc.ReLU()[0], activateFunc.ReLU()[1], "ReLU")
    PlotMultiFunc(x, activateFunc.LeakyReLU()[0], activateFunc.LeakyReLU()[1], "LeakyReLU")
    PlotMultiFunc(x, activateFunc.PReLU()[0], activateFunc.PReLU()[1], "PReLU")
    PlotMultiFunc(x, activateFunc.ReLU6()[0], activateFunc.ReLU6()[1], "ReLU6")
    PlotMultiFunc(x, activateFunc.ELU()[0], activateFunc.ELU()[1], "ELU")
    PlotMultiFunc(x, activateFunc.SELU()[0], activateFunc.SELU()[1], "SELU")
    PlotMultiFunc(x, activateFunc.Swish()[0], activateFunc.Swish()[1], "Swish")
    PlotMultiFunc(x, activateFunc.Hard_Swish()[0], activateFunc.Hard_Swish()[1], "Hard_Swish")
    PlotMultiFunc(x, activateFunc.Mish()[0], activateFunc.Mish()[1], "Mish")
    PlotMultiFunc(x, activateFunc.GELU()[0], activateFunc.GELU()[1], "GELU")
