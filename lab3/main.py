import numpy as np
from optimization import subgradient_method, proximal_gradient_method, proximal_fast_gradient_method
from oracles import create_lasso_nonsmooth_oracle, create_lasso_prox_oracle
import matplotlib.pyplot as plt

def experiment_1():
    plt.clf()

    m = 10
    n = 5

    A = np.random.rand(m, n)
    b = np.random.rand(m)
    regcoef = 1
    #print(A)
    #print(b)
    oracle = create_lasso_nonsmooth_oracle(A, b, regcoef)

    x_0s = [
        np.zeros(n),
        np.random.rand(n),
        np.ones(n),
        np.ones(n) * 10,
        np.ones(n) * 100,
    ]

    #alpha_0s = [alpha for alpha in range(1, 11)]
    alpha_0s = [alpha / 3 for alpha in range(1, 10)]

    for i, x_0 in enumerate(x_0s):
        iteraions_num = []
        for alpha_0 in alpha_0s:
            _, _, history = subgradient_method(oracle, x_0, max_iter=10 ** 5, alpha_0=alpha_0, trace=True)
            iteraions_num.append(len(history['time']))
        plt.plot(alpha_0s, iteraions_num, label = f"{i + 1}")

    plt.legend()
    plt.xlabel('alpha_0')
    plt.ylabel('number of iterations')
    plt.savefig('1/2.png')


def experiment_2():
    plt.clf()

    m = 10
    n = 5

    A = np.random.rand(m, n)
    b = np.random.rand(m)
    regcoef = 1

    oracle = create_lasso_prox_oracle(A, b, regcoef)
    x_0 = np.random.rand(n)

    _, _, grad_hist = proximal_gradient_method(oracle, x_0, trace=True)
    _, _, fast_grad_hist = proximal_fast_gradient_method(oracle, x_0, trace=True)

    x = [i for i in range(1, len(grad_hist['iterations']) + 1)]
    y = grad_hist['iterations']
    plt.plot(x, y, label = 'prox grad method')

    mean_y = np.mean(y)
    means = [mean_y for _ in range(len(x))]
    plt.plot(x, means, label = f'mean number of iterations = {mean_y}')

    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('line search iterations')
    plt.savefig('2/grad.png')

    plt.clf()

    x = [i for i in range(1, len(fast_grad_hist['iterations']) + 1)]
    y = fast_grad_hist['iterations']
    plt.plot(x, y, label = 'prox fast grad method')

    mean_y = np.mean(y)
    means = [mean_y for _ in range(len(x))]
    plt.plot(x, means, label = f'mean number of iterations = {mean_y}')
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('line search iterations')
    plt.savefig('2/fast_grad.png')

def experiment_3():
    ns = [
        10,
        100,
        1000
    ]

    ms = [
        100,
        1000,
        10000
    ]

    regcoefs = [
        0.01,
        0.1,
        1,
        10
    ]


    for n in ns:
        x_0 = np.random.rand(n)
        for m in ms:
            for regcoef in regcoefs:
                plt.clf()
                label = f"n={n}_m={m}_regcoef={regcoef}"
                A = np.random.rand(m, n)
                b = np.random.rand(m)

                oracle_1 = create_lasso_nonsmooth_oracle(A, b, regcoef)
                oracle_2 = create_lasso_prox_oracle(A, b, regcoef)

                max_iter = 1000
                _, _, subgrad = subgradient_method(oracle_1, x_0, max_iter, trace=True)
                _, _, grad = proximal_gradient_method(oracle_2, x_0, max_iter=max_iter, trace=True)
                _, _, fast_grad = proximal_fast_gradient_method(oracle_2, x_0, max_iter=max_iter, trace=True)

                x = [i + 1 for i in range (len(subgrad['duality_gap']))]
                y = subgrad['duality_gap']
                plt.yscale("log")
                plt.plot(x, y, label = 'subgradient_method')

                x = [i + 1 for i in range (len(grad['duality_gap']))]
                y = grad['duality_gap']
                plt.yscale("log")
                plt.plot(x, y, label = 'gradient_method')

                x = [i + 1 for i in range (len(fast_grad['duality_gap']))]
                y = fast_grad['duality_gap']
                plt.yscale("log")
                plt.plot(x, y, label = 'fast gradient_method')

                plt.legend()
                plt.xlabel('number of iterations')
                plt.ylabel('duality_gap in log scale')
                plt.savefig(f'3/a/{label}.png')

                plt.clf()

                x = subgrad['time']
                y = subgrad['duality_gap']
                plt.yscale("log")
                plt.plot(x, y, label = 'subgradient_method')

                x = grad['time']
                y = grad['duality_gap']
                plt.yscale("log")
                plt.plot(x, y, label = 'gradient_method')

                x = fast_grad['time']
                y = fast_grad['duality_gap']
                plt.yscale("log")
                plt.plot(x, y, label = 'fast gradient_method')

                plt.legend()
                plt.xlabel('time')
                plt.ylabel('duality_gap in log scale')

                plt.savefig(f'3/b/{label}.png')








def main():
    np.random.seed(42)
    #experiment_1()
    #experiment_2()
    experiment_3()

if __name__ == "__main__":
    main()