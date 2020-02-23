import numpy as np
from HW1.gradient_descent import gradient_descent, linear_step_chooser
from HW1.one_demensional import golden
from scipy.special import expit


# TODO: make it work
class Logit:
    def __init__(self, alpha):
        self.alpha = alpha
        self.w = None

    def fit(self, X, y):
        objects_count, features_count = X.shape
        assert y.shape == (objects_count,)
        ones = np.ones((objects_count, 1))
        X_r = np.hstack((X, ones))

        start_w = np.ones((features_count + 1, 1))

        def Q(weights):
            answers = np.matmul(X_r, weights)
            margins = answers * y
            return np.sum(np.logaddexp(0, -margins)) / objects_count

        A = np.zeros((features_count + 1, objects_count))
        for j in range(features_count + 1):
            for i in range(objects_count):
                A[j, i] = -y[i] * X_r[i, j]

        def Q_grad(weights):
            answers = np.matmul(X_r, weights)
            margins = answers * y
            b = expit(-margins)
            return np.matmul(A, b) / objects_count

        self.w = gradient_descent(Q, Q_grad, start_w, linear_step_chooser(golden), 'gradient')


def main():
    logit = Logit(1.)
    logit.fit(np.array([[1., 2., 3.], [4., 5., 6.]]), np.array([1., -1.]))
    print(logit.w)


if __name__ == '__main__':
    main()
