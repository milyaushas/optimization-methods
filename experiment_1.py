import matplotlib.pyplot as plt
import numpy as np
from oracles import QuadraticOracle
from optimization import gradient_descent, conjugate_gradients

def experiment_1():
    #plot(optimization_method="gradient_descent", path="./1/grad_descent_tmp.png")
    #plot(optimization_method="gradient_descent", method="Constant", path="./1/grad_descent_Constant.png")
    #plot(optimization_method="gradient_descent", method="Armijo",   path="./1/grad_descent_Armijo_small.png")
    #plot(optimization_method="gradient_descent", method="Wolfe",    path="./1/grad_descent_Wolfe_small.png")

    #plot(optimization_method="conjugate_gradients", path="./1/conjugate_gradients.png")
    plot(optimization_method="conjugate_gradients", path="./1/conjugate_gradients_small.png")


def generate_Ab(k, n):
    a = np.random.uniform(low=1.0, high=k, size=n)
    min_ind = np.random.randint(0, n - 1)
    max_ind = np.random.randint(0, n - 1)
    while max_ind == min_ind:
        max_ind = np.random.randint(0, n - 1)
    a[min_ind] = 1
    a[max_ind] = k
    A = np.eye(n) * a
    b = np.random.uniform(-10000, 10000, n)
    return A, b


def plot(optimization_method="gradient_descent", method="Constant", path="./tmp.png"):
    ns = [10, 100, 1000]
    ks = np.array(list(range(1, 1101, 100)))
    iterations = 10

    colors = {10: "red",
              100: "blue",
              1000: "green",
              10000: "pink"
              }

    plt.clf()
    for n in ns:
        # print(f"N = {n}")
        opts = {"method": "Constant", "c": "0.01"}
        if method == "Armijo":
            opts = {"method": "Armijo"}
        if method == "Wolfe":
            opts = {"method": "Wolfe"}

        def find_T(k):
            A, b = generate_Ab(k, n)
            oracle = QuadraticOracle(A, b)
            x_0 = np.random.uniform(-10000, 10000, n)
            if optimization_method == "gradient_descent":
                _, _, history = gradient_descent(oracle, x_0=x_0, line_search_options=opts, trace=True, display=True)
            else:
                _, _, history = conjugate_gradients(lambda x: A.dot(x), b, x_0, max_iter=10000, trace=True, display=True)
            return len(history['time'])

        ts_all = np.zeros(shape=(iterations, len(ks)))
        for i in range(iterations):
            vfunc = np.vectorize(find_T)
            ts_all[i] = vfunc(ks)
            plt.plot(ks, ts_all[i], "--", color=colors[n])
        plt.plot(ks, ts_all.mean(axis=0), color=colors[n])


    plt.xlabel('k - число обусловленности')
    plt.ylabel('T(n, k) - число итераций до сходимости')
    plt.grid()
    plt.savefig(path)
