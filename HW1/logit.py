import numpy as np
from HW1.gradient_descent import gradient_descent, linear_step_chooser
from HW1.one_demensional import golden
from scipy.special import expit
import math


# TODO: use numpy stuff to make it faster, add newton solver
class Logit:
    def __init__(self, alpha, solver='gradient'):
        assert solver in {'gradient'}
        self.alpha = alpha
        self.w = None
        self.solver = solver

    @staticmethod
    def __add_feature(X):
        objects_count, _ = X.shape
        ones = np.ones((objects_count, 1))
        return np.hstack((X, ones))

    def fit(self, X, y):
        objects_count, features_count = X.shape
        assert y.shape == (objects_count,)
        X_r = Logit.__add_feature(X)

        start_w = np.random.normal(loc=0., scale=1., size=features_count + 1)

        def Q(weights):
            result = 0.
            for i in range(objects_count):
                cur_object = X_r[i]
                cur_prediction = np.dot(cur_object, weights)
                cur_answer = y[i]
                result += math.log(1 + math.exp(-cur_prediction * cur_answer))
            result /= objects_count
            norm = 0.
            for i in range(features_count + 1):
                norm += weights[i] ** 2
            norm *= self.alpha / 2
            return result + norm

        def Q_grad(weights):
            grad = np.zeros(features_count + 1)
            for j in range(features_count + 1):
                for i in range(objects_count):
                    cur_object = X_r[i]
                    cur_answer = y[i]
                    cur_prediction = np.dot(cur_object, weights)
                    grad[j] -= (cur_answer * cur_object[j]) / (1 + math.exp(cur_prediction * cur_answer))
            grad /= objects_count
            return grad + self.alpha * weights

        self.w = gradient_descent(Q, Q_grad, start_w, lambda x, y, z: 0.01, 'grad', eps=1e-9)[-1]

    def predict(self, X):
        X_r = Logit.__add_feature(X)
        return np.sign(np.matmul(X_r, self.w)).astype(int)
