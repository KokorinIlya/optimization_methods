import numpy as np
from HW1.gradient_descent import gradient_descent, linear_step_chooser
from HW1.one_demensional import golden
from scipy.special import expit


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


# TODO: use numpy stuff to make it faster
class Logit:
    def __init__(self, alpha):
        self.alpha = alpha
        self.w = None

    def fit(self, X, y):
        objects_count, features_count = X.shape
        assert y.shape == (objects_count,)
        ones = np.ones((objects_count, 1))
        X_r = np.hstack((X, ones))

        start_w = np.zeros(features_count + 1)

        def Q(weights):
            result = 0.
            for i in range(objects_count):
                cur_object = X_r[i]
                cur_prediction = np.dot(cur_object, weights)
                cur_answer = y[i]
                result += np.logaddexp(0, -cur_prediction * cur_answer)
            result /= objects_count
            norm = 0.
            for i in range(features_count + 1):
                norm += weights[i] ** 2
            result += norm * self.alpha / 2
            return result

        A = np.zeros((features_count + 1, objects_count))
        for j in range(features_count + 1):
            for i in range(objects_count):
                A[j, i] = -y[i] * X_r[i, j]

        def Q_grad(weights):
            b = np.zeros(objects_count)
            for i in range(objects_count):
                cur_object = X_r[i]
                cur_prediction = np.dot(cur_object, weights)
                cur_answer = y[i]
                cur_margin = cur_prediction * cur_answer
                b[i] = expit(-cur_margin)
            grad = np.matmul(A, b) / objects_count
            return grad + self.alpha * weights

        self.w = gradient_descent(Q, Q_grad, start_w, linear_step_chooser(golden), 'grad')[-1]
