import numpy as np
from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from utils import get_line_search_tool
import time


def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    # TODO: Implement Conjugate Gradients method.

    result_message = 'success'

    if max_iter is None:
        max_iter = len(x_0)

    start_time = time.time()

    iter = 0
    Ax_k = matvec(x_k)
    g_k = Ax_k - b
    d_k = -g_k

    while iter < max_iter:
        iter += 1

        Ad_k = matvec(d_k)
        alpha_k = np.dot(g_k, g_k) / np.dot(d_k, Ad_k)
        x_k += alpha_k * d_k
        g_k_1 = g_k + alpha_k * Ad_k

        if np.linalg.norm(matvec(x_k) - b) <= tolerance * np.linalg.norm(b):
            break

        beta_k = np.dot(g_k_1, g_k_1) / np.dot(g_k, g_k)
        d_k = (-1) * g_k_1 + beta_k * d_k
        g_k = g_k_1

        if trace:
            history['time'].append(time.time() - start_time)
            history['residual_norm'].append(np.linalg.norm(g_k_1, ord=2))
            if x_k.size <= 2:
                history['x'].append(x_k)

    if np.linalg.norm(matvec(x_k) - b) > tolerance * np.linalg.norm(b):
        result_message = 'iterations_exceeded'

    if display:
        print(result_message)

    return x_k, result_message, history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    # TODO: Implement L-BFGS method.
    # Use line_search_tool.line_search() for adaptive step size.
    n = len(x_0)
    iter = 0
    H = deque()
    g_k = oracle.grad(x_0)

    def BFGS_multiply(v, H_k, gamma_0):
        if len(H_k) == 0:
            return gamma_0 * v
        s, y = H_k[-1]
        H_k.pop()
        v_ = v - (np.dot(s, v))/(np.dot(y, s)) * y
        z = BFGS_multiply(v_, H_k, gamma_0)
        return z + (np.dot(s, v) - np.dot(y, z)) / (np.dot(y, s)) * s

    def LBFGS_direction():
        if len(H) == 0:
            return -oracle.grad(x_k)
        s, y = H[-1]
        gamma_0 = (np.dot(y, s)) / (np.dot(y, y))
        H_k = H.copy()
        return BFGS_multiply(-oracle.grad(x_k), H_k,  gamma_0)

    grad_x0_norm = np.linalg.norm(g_k)
    start_time = time.time()
    try:
        while iter < max_iter:
            iter += 1
            d_k = LBFGS_direction()
            x_k_1 = x_k
            alpha_k = get_line_search_tool().line_search(oracle, x_k, d_k) #Wolfe
            x_k = x_k + alpha_k * d_k
            g_k_1 = g_k
            g_k = oracle.grad(x_k)
            if trace:
                history['func'].append(oracle.func(x_k))
                history['time'].append(time.time() - start_time)
                history['grad_norm'].append(np.linalg.norm(g_k))
                if n <= 2:
                    history['x'].append(x_k)

            if np.linalg.norm(g_k) <= tolerance * grad_x0_norm:
                break
            s_k = x_k - x_k_1
            y_k = g_k - g_k_1
            H.append((s_k, y_k))
            if len(H) > memory_size:
                H.popleft()
    except Exception as e:
        if display:
            print(str(e))
        return None, 'computational_error', history

    if iter == max_iter and np.linalg.norm(oracle.grad(x_k)) > tolerance * grad_x0_norm:
        return None, 'iterations_exceeded', history

    return x_k, 'success', history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500,
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_options = {"method": "Armijo", "alpha_0": 1.0}
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    # TODO: Implement hessian-free Newton's method.
    # Use line_search_tool.line_search() for adaptive step size.
    result_message = "success"
    start_time = time.time()
    iter = 0
    g_k = oracle.grad(x_0)
    d_k_1 = -g_k
    try:
        while iter < max_iter:
            iter += 1
            eps_k = min(0.5, np.sqrt(np.linalg.norm(g_k))) * np.linalg.norm(g_k)
            matvec = lambda v: oracle.hess_vec(x_k, v)
            d_k, message, _ = conjugate_gradients(matvec, -g_k, d_k_1, eps_k)
            while message != "success" or g_k.dot(d_k) >= 0:
                if display and message != "success":
                    print("Something went wrong while performing conjugate_gradients methods:", message)
                eps_k /= 10
                d_k, message, _ = conjugate_gradients(matvec, -g_k, d_k, eps_k)
            alpha_k = line_search_tool.line_search(oracle, x_k, d_k)
            x_k = x_k + alpha_k * d_k
            g_k = oracle.grad(x_k)
            if trace:
                history['func'].append(oracle.func(x_k))
                history['time'].append(time.time() - start_time)
                history['grad_norm'].append(np.linalg.norm(g_k))
                if len(x_0) <= 2:
                    history['x'].append(x_k)
            # как на лекции
            if np.linalg.norm(g_k) ** 2 <= tolerance:
                break
            d_k_1 = d_k
    except Exception as e:
        if display:
            print(str(e))
        result_message = 'computational_error'
    if result_message != 'success':
        return None, result_message, history

    if np.linalg.norm(g_k) ** 2 > tolerance and iter == max_iter:
        result_message = "iterations_exceeded"
    return x_k, result_message, history


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradient descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    # TODO: Implement gradient descent
    # Use line_search_tool.line_search() for adaptive step size.er:
    start_time = time.time()
    it = 0
    alpha = None
    result_message = "success"

    try:
        while it < max_iter and np.linalg.norm(oracle.grad(x_k), ord=2) ** 2 > tolerance * (
                np.linalg.norm(oracle.grad(x_0), ord=2) ** 2):
            d_k = -oracle.grad(x_k)
            alpha = line_search_tool.line_search(oracle, x_k, d_k, alpha)
            if alpha is None:
                result_message = 'computational_error'
                break
            x_k += float(alpha) * d_k
            if trace:
                history['time'].append(time.time() - start_time)
                history['func'].append(oracle.func(x_k))
                history['grad_norm'].append(np.linalg.norm(oracle.grad(x_k), ord=2))
                if len(x_k) <= 2:
                    history['x'].append(x_k)
            # if display:
            # print(f"iter={it}, x_k={x_k}, f(x_k)={oracle.func(x_k)}")
            it += 1
    except Exception as e:
        if display:
            print(str(e))
        result_message = 'computational_error'

    if it == max_iter and np.linalg.norm(oracle.grad(x_k)) ** 2 > tolerance * (np.linalg.norm(oracle.grad(x_0)) ** 2):
        result_message = 'iterations_exceeded'

    if display:
        print(result_message)

    return x_k, result_message, history
