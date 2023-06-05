import numpy as np
import scipy
from scipy.special import expit



class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A

    def minimize_directional(self, x, d):
        """
        Minimizes the function with respect to a specific direction:
            Finds alpha = argmin f(x + alpha d)
        """
        # TODO: Implement for bonus part
        pass


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATy : function of y
            Computes matrix-vector product A^Ty, where y is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        # TODO: Implement
        return np.mean(
            np.logaddexp(np.zeros_like(self.b), - self.b * self.matvec_Ax(x))) + self.regcoef / 2 * np.linalg.norm(
            x) ** 2

    def grad(self, x):
        # TODO: Implement
        m = len(self.b)
        tmp = (scipy.special.expit(self.b * self.matvec_Ax(x)) - np.ones(m)) * self.b
        return 1 / m * self.matvec_ATx(tmp) + self.regcoef * x

    def hess(self, x):
        # TODO: Implement
        n = len(x)
        m = len(self.b)
        g = scipy.special.expit(-self.b * self.matvec_Ax(x))
        return self.matmat_ATsA(g * (1.0 - g)) / m + self.regcoef * np.eye(n)

    def hess_vec(self, x, v):
        # TODO: Implement
        return self.hess(x).dot(v)


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return None

    def grad_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return None


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """

    if isinstance(A, scipy.sparse.csr_matrix):
        mat_A = A.toarray()
    elif isinstance(A, np.ndarray):
        mat_A = A
    else:
        raise ValueError("A must be an object of np.ndarray or scipy.sparse.csr_matrix")

    matvec_Ax = lambda x: mat_A.dot(x)  # TODO: Implement
    matvec_ATx = lambda x: mat_A.T.dot(x)  # TODO: Implement

    def matmat_ATsA(s):
        # TODO: Implement
        return A.T.dot(np.diag(s)).dot(A)

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def hess_vec_finite_diff(func, x, v, eps=1e-5):
    """
    Returns approximation of the matrix product 'Hessian times vector'
    using finite differences.
    """
    # TODO: Implement numerical estimation of the Hessian times vector
    n = len(x)
    E = np.eye(n)
    hess_vec = np.zeros(n)
    for i, e_i in enumerate(E):
        hess_vec[i] = (func(x + eps * v + eps * e_i) - func(x + eps * v) - func(x + eps * e_i) + func(x)) / eps**2
    return hess_vec
def test_hess_vec():
    A = np.random.rand(3, 3)
    b = np.random.rand(3)
    oracle = create_log_reg_oracle(A, b, 1)
    x = np.random.rand(3)
    v = np.random.rand(3)
    print(oracle.hess_vec(x, v))
    print(hess_vec_finite_diff(oracle.func, x, v))


def perform_test_multiple_times(n=5):
    for i in range(n):
        test_hess_vec()