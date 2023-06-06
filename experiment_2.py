import numpy as np
from optimization import lbfgs
from oracles import create_log_reg_oracle
import sklearn.datasets as ds
import matplotlib.pyplot as plt


def experiment_2():
    print("loading started")
    A, b = ds.load_svmlight_file('./datasets/gisette_scale')
    print("loading ended")
    m = len(b)
    print(m)
    oracle = create_log_reg_oracle(A, b, 1 / m)

    ls = [1, 5, 10, 50, 100]
    colours = {
        1: "red",
        5: "orange",
        10: "green",
        50: "blue",
        100: "purple"
    }
    x_0 = np.zeros(A.shape[1])

    histories = dict()

    print("perform lbfgs")
    for l in ls:
        print("l = ", l)
        _, message, history = lbfgs(oracle, x_0, memory_size=l, trace=True, display=True, max_iter=10000)
        print(message)
        histories[l] = history

    grad_x0_norm_squared = np.linalg.norm(oracle.grad(x_0))**2

    print("plotting graphs")

    for l in ls:
        print("l = ", l)
        xs = range(len(histories[l]['grad_norm'])) # номера итераций
        vectorized_func = np.vectorize(lambda x: np.log(x ** 2 / grad_x0_norm_squared))
        ys = vectorized_func(histories[l]['grad_norm'])
        plt.plot(xs, ys, color=colours[l])
    plt.legend(('l = 1', 'l = 5', 'l = 10', 'l = 50', 'l = 100'))
    plt.grid()
    plt.xlabel("Номер итерации")
    plt.ylabel("||grad(x_k)||^2 / ||grad(x_0)||^2 в логарифмической шкале")
    plt.savefig("./2/a.png")


    plt.clf()
    for l in ls:
        print("l = ", l)
        xs = histories[l]['time']
        vectorized_func = np.vectorize(lambda x: np.log(x ** 2 / grad_x0_norm_squared))
        ys = vectorized_func(histories[l]['grad_norm'])
        plt.plot(xs, ys, color=colours[l])
    plt.legend(('l = 1', 'l = 5', 'l = 10', 'l = 50', 'l = 100'))
    plt.grid()
    plt.xlabel("Время в секундах")
    plt.ylabel("||grad(x_k)||^2 / ||grad(x_0)||^2 в логарифмической шкале")
    plt.savefig("./2/b.png")
