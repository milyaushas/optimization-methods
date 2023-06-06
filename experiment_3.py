import numpy as np
from optimization import hessian_free_newton, lbfgs, gradient_descent
from oracles import create_log_reg_oracle
import sklearn.datasets as ds
import matplotlib.pyplot as plt


def experiment_3():
    datasets = ["gisette_scale", "real-sim", "news20.binary", "rcv1_train.binary"]
    for dataset in datasets:
        print("loading started")
        A, b = ds.load_svmlight_file(f'./datasets/{dataset}')
        print("loading ended")
        n = A.shape[1]
        m = len(b)
        print(m)
        oracle = create_log_reg_oracle(A, b, 1 / m)
        x_0 = np.zeros(n)

        methods = ['HFN', 'L-BFGS', 'GD']
        histories = dict()
        colours = {
            'HFN': 'red',
            'L-BFGS': 'blue',
            'GD': 'pink'
        }

        _, message, histories['HFN'] = hessian_free_newton(oracle, x_0, trace=True)
        print(message)
        _, message, histories['L-BFGS'] = lbfgs(oracle, x_0, trace=True)
        print(message)
        _, message, histories['GD'] = gradient_descent(oracle, x_0, trace=True)
        print(message)

        # Зависимость значения функции против номера итерации метода.
        for method in methods:
            print(method)
            xs = range(len(histories[method]['func']))  # номера итераций
            ys = histories[method]['func']
            plt.plot(xs, ys, colours[method])
        plt.legend(('HFN', 'L-BFGS', 'GD'))
        plt.grid()
        plt.xlabel('Номер итерации')
        plt.ylabel('Значение функции')
        plt.savefig(f"./3/3a_{dataset}.png")
        plt.clf()

        # Зависимость значения функции против номера итерации метода.
        for method in methods:
            print(method)
            xs = histories[method]['time']
            ys = histories[method]['func']
            plt.plot(xs, ys, colours[method])
        plt.legend(('HFN', 'L-BFGS', 'GD'))
        plt.grid()
        plt.xlabel('Время работы в секундах')
        plt.ylabel('Значение функции')
        plt.savefig(f"./3/3b_{dataset}.png")
        plt.clf()

        # Зависимость относительного квадрата нормы градиента
        grad_x0_norm_squared = np.linalg.norm(oracle.grad(x_0)) ** 2
        for method in methods:
            print(method)
            xs = histories[method]['time']
            vectorized_func = np.vectorize(lambda x: np.log(x ** 2 / grad_x0_norm_squared))
            ys = vectorized_func(histories[method]['grad_norm'])
            plt.plot(xs, ys, colours[method])
        plt.legend(('HFN', 'L-BFGS', 'GD'))
        plt.grid()
        plt.xlabel('Время работы в секундах')
        plt.ylabel('||grad(x_k)||^2 / ||grad(x_0)||^2 в логарифмической шаколе')
        plt.savefig(f"./3/3c_{dataset}.png")

experiment_3()
