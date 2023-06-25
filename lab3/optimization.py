from collections import defaultdict
import numpy as np
from numpy.linalg import norm, solve
from time import time
import datetime


def subgradient_method(oracle, x_0, tolerance=1e-2, max_iter=1000, alpha_0=1,
                       display=False, trace=False):
    """
    Subgradient descent method for nonsmooth convex optimization.

    Parameters
    ----------
    oracle : BaseNonsmoothConvexOracle-descendant object
        Oracle with .func() and .subgrad() methods implemented for computing
        function value and its one (arbitrary) subgradient respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    alpha_0 : float
        Initial value for the sequence of step-sizes.
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
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    # TODO: implement.
    start_time = time()

    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0) * 1.0
    x_star = np.copy(x_0)
    f_min = oracle.func(x_k)
    f_k = f_min
    duality_gap = oracle.duality_gap(x_k)

    # we can check stopping  criterion only if .duality_gap() is available in oracle
    CHECK_STOPPING_CRITERION = duality_gap is not None


    for k in range(1, max_iter + 1):
        if trace:
            history['func'].append(f_k)
            history['time'].append(time() - start_time)
            if CHECK_STOPPING_CRITERION:
                history['duality_gap'].append(duality_gap)
            if x_k.size <= 2:
                history['x'].append(x_k)

        if CHECK_STOPPING_CRITERION:
            if np.abs(duality_gap) <= tolerance:
                return x_k, 'success', history


        if k == max_iter and CHECK_STOPPING_CRITERION:
            return x_star, 'iterations_exceeded', history

        alpha_k = alpha_0 / np.sqrt(k + 1)
        subgrad = oracle.subgrad(x_k)
        x_k = x_k - alpha_k * subgrad / np.linalg.norm(subgrad)
        f_k = oracle.func(x_k)
        duality_gap = oracle.duality_gap(x_k)

        if f_k < f_min:
            f_min = f_k
            x_star = np.copy(x_k)

        if display:
            print(f"k = {k}, x_star = {x_star}, f_min = {f_min}")

    return x_star, 'success', history


def proximal_gradient_method(oracle, x_0, L_0=1, tolerance=1e-5,
                              max_iter=1000, trace=False, display=False):
    """
    Gradient method for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented 
        for computing function value, its gradient and proximal mapping 
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
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
            - history['func'] : list of objective function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    # TODO: implement.
    start_time = time()
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0) * 1.0
    duality_gap = oracle.duality_gap(x_k)
    L_k = L_0
    f_k = oracle.func(x_k)

    # we can check stopping  criterion only if .duality_gap() is available in oracle
    CHECK_STOPPING_CRITERION = duality_gap is not None

    k = 0

    while k <= max_iter:
        if display:
            print(f"k = {k}, x_k = {x_k}, f_k= {f_k}")

        if trace:
            history['func'].append(f_k)
            history['time'].append(time() - start_time)
            if CHECK_STOPPING_CRITERION:
                history['duality_gap'].append(duality_gap)
            if x_k.size <= 2:
                history['x'].append(x_k)

        if CHECK_STOPPING_CRITERION:
            if np.abs(duality_gap) <= tolerance:
                return x_k, 'success', history

        if k == max_iter:
            if CHECK_STOPPING_CRITERION:
                return x_k, 'iterations_exceeded', history
            return x_k, 'success', history

        grad = oracle.grad(x_k)

        x_next = oracle.prox(x_k - 1.0 / L_k * grad, 1.0 / L_k)

        x_diff = x_next - x_k
        if oracle.func(x_next) <= f_k + np.dot(grad, x_diff) + L_k / 2.0 * np.dot(x_diff, x_diff):
            L_k = L_k / 2.0
            f_k = oracle.func(x_next)
            x_k = x_next
            duality_gap = oracle.duality_gap(x_k)
            k += 1
            continue

        L_k = 2.0 * L_k


def proximal_fast_gradient_method(oracle, x_0, L_0=1.0, tolerance=1e-5,
                              max_iter=1000, trace=False, display=False):
    """
    Fast gradient method for composite minimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented 
        for computing function value, its gradient and proximal mapping 
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
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
            - history['func'] : list of objective function values phi(best_point) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
    """
    # TODO: Implement
    start_time = time()
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0) * 1.0
    duality_gap = oracle.duality_gap(x_k)
    L_k = L_0
    A_k  = 0
    v_k = np.copy(x_0)
    f_k = oracle.func(x_k)

    x_star = np.copy(x_k)
    f_min = np.copy(f_k)

    # we can check stopping  criterion only if .duality_gap() is available in oracle
    CHECK_STOPPING_CRITERION = duality_gap is not None

    k = 0
    s = 0

    while k <= max_iter:
        if display:
            print(f"k = {k}, x_k = {x_k}, f_k = {f_k}, x_start = {x_star}, f_min = {f_min}")

        if trace:
            history['func'].append(f_k)
            history['time'].append(time() - start_time)
            if CHECK_STOPPING_CRITERION:
                history['duality_gap'].append(duality_gap)
            if x_k.size <= 2:
                history['x'].append(x_k)

        if CHECK_STOPPING_CRITERION:
            if np.abs(duality_gap) <= tolerance:
                return x_k, 'success', history

        if k == max_iter:
            if CHECK_STOPPING_CRITERION:
                return x_k, 'iterations_exceeded', history
            return x_k, 'success', history

        a_k = (1 + np.sqrt(1 + 4 * L_k * A_k)) / (2.0 * L_k)
        A_next = A_k + a_k
        y_k = (A_k * x_k + a_k * v_k) / A_next

        s_k = s +  a_k * oracle.grad(y_k)
        v_next = oracle.prox(x_0 - s)

        x_next = (A_k * x_k + a_k * v_next) / A_next

        diff = x_next - y_k
        f_next = oracle.func(x_next)
        if f_next > oracle.func(y_k) + np.dot(oracle.grad(y_k), diff) + L_k / 2.0 * np.dot(diff, diff):
            L_k = 2.0 * L_k
            continue

        L_k = L_k / 2
        x_k = x_next
        v_k = v_next
        f_k = f_next
        s = s_k
        k += 1

        if f_k < f_min:
            x_star = x_k
            f_min = f_k
