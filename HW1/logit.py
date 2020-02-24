import numpy as np
from HW1.gradient_descent import gradient_descent, linear_step_chooser
from HW1.newton_method import newton
from HW1.one_demensional import golden
from scipy.special import expit
import math


# TODO: use numpy stuff to make it faster
class Logit:
    def __init__(self, alpha, solver, max_errors=100):
        assert solver in {'gradient', 'newton'}
        self.alpha = alpha
        self.w = None
        self.solver = solver
        self.max_errors = max_errors

    @staticmethod
    def __add_feature(X):
        objects_count, _ = X.shape
        ones = np.ones((objects_count, 1))
        return np.hstack((X, ones))

    def fit(self, X, y, debug_iters=None):
        objects_count, features_count = X.shape
        assert y.shape == (objects_count,)
        X_r = Logit.__add_feature(X)

        start_w = np.random.normal(loc=0., scale=1., size=features_count + 1)

        def Q(weights):
            predictions = np.matmul(X_r, weights)
            margins = predictions * y
            losses = np.logaddexp(0, -margins)
            return (np.sum(losses) / objects_count) + (np.sum(weights ** 2) * self.alpha / 2)

        A = np.transpose(X_r * y.reshape((objects_count, 1)))

        def Q_grad(weights):
            predictions = np.matmul(X_r, weights)
            margins = predictions * y
            b = expit(-margins)
            grad = -np.matmul(A, b) / objects_count
            return grad + self.alpha * weights

        def Q_hess(weights):
            hess = np.zeros((features_count + 1, features_count + 1))
            for j in range(features_count + 1):
                for k in range(features_count + 1):
                    for i in range(objects_count):
                        cur_object = X_r[i]
                        cur_answer = y[i]
                        cur_prediction = np.dot(cur_object, weights)
                        cur_exp = math.exp(cur_prediction * cur_answer)
                        hess[j, k] += (cur_object[j] * cur_object[k] * cur_exp) / ((cur_exp + 1) ** 2)
            hess /= objects_count
            return hess + self.alpha * np.eye(features_count + 1)

        if self.solver == 'gradient':
            # TODO: fastest descent
            self.w = gradient_descent(Q, Q_grad, start_w, linear_step_chooser(golden), 'grad', eps=1e-5,
                                      debug_iters=debug_iters)[-1]
        else:
            errors = 0
            while True:
                try:
                    if errors >= self.max_errors:
                        self.w = start_w
                    else:
                        self.w = newton(Q, Q_grad, Q_hess, start_w, 'delta', eps=1e-9)[-1]
                    break
                except ArithmeticError:
                    errors += 1
                    start_w = np.random.normal(loc=0., scale=1., size=features_count + 1)

    def predict(self, X):
        X_r = Logit.__add_feature(X)
        return np.sign(np.matmul(X_r, self.w)).astype(int)
