import numpy as np
from HW1.gradient_descent import gradient_descent
from math import sqrt


def create_matrix(condition_number, n):
    r = sqrt(condition_number)
    A = np.random.randn(n, n)
    u, s, v = np.linalg.svd(A)
    h, l = np.max(s), np.min(s)  # highest and lowest eigenvalues (h / l = current cond number)
    # linear stretch: f(s) = a * s + b, f(h) = h, f(l) = h/k
    new_s = h * (1 - ((r - 1) / r) / (h - l) * (h - s))
    new_A = (u * new_s) @ v.T  # make inverse transformation (here cond number is sqrt(k))
    new_A = new_A @ new_A.T  # make matrix symmetric and positive semi-definite (cond number is just k)
    assert np.isclose(np.linalg.cond(new_A), condition_number)
    return new_A


def number_of_iters(cond, n_vars, step_chooser, n_checks=100):
    avg_iters = 0
    for _ in range(n_checks):
        A = create_matrix(cond, n_vars)
        b = np.random.randn(len(A))
        init_x = np.random.randn(len(A))
        f = lambda x: x.dot(A).dot(x) - b.dot(x)
        f_grad = lambda x: (A + A.T).dot(x) - b

        # print(f(np.linalg.inv(A+A.T).dot(b))) -- optimal value

        trace = gradient_descent(f, f_grad, init_x, step_chooser, 'value')

        avg_iters += len(trace)
    return avg_iters / n_checks
