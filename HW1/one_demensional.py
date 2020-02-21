import math


def dichotomy(f, left_border, right_border, eps=1e-5):
    iterations = 0
    delta = eps / 2
    while right_border - left_border > eps:
        iterations += 1
        middle = (left_border + right_border) / 2
        x_left = middle - delta
        x_right = middle + delta

        f_left = f(x_left)
        f_right = f(x_right)

        if f_left < f_right:
            right_border = x_right
        elif f_left > f_right:
            left_border = x_left
        else:
            return middle, iterations
    return (left_border + right_border) / 2, iterations


def golden(f, left_border, right_border, eps=1e-5):
    phi = (1 + math.sqrt(5)) / 2
    iterations = 0

    interval_len = right_border - left_border
    x_left = left_border + (2 - phi) * interval_len
    x_right = right_border - (2 - phi) * interval_len
    f_left = f(x_left)
    f_right = f(x_right)

    while interval_len > eps:
        iterations += 1
        if f_left < f_right:
            right_border = x_right
            x_right = x_left
            f_right = f_left
            interval_len = right_border - left_border
            x_left = left_border + (2 - phi) * interval_len
            f_left = f(x_left)
        elif f_left > f_right:
            left_border = x_left
            x_left = x_right
            f_left = f_right
            interval_len = right_border - left_border
            x_right = right_border - (2 - phi) * interval_len
            f_right = f(x_right)
        else:
            return (right_border + left_border) / 2, iterations
    return (right_border + left_border) / 2, iterations


def fib(f, left_border, right_border, n):
    fibs = [1, 1]
    while len(fibs) < n + 1:
        fibs.append(fibs[-1] + fibs[-2])

    interval_len = right_border - left_border
    x_left = left_border + (fibs[n - 2] / fibs[n]) * interval_len
    x_right = left_border + (fibs[n - 1] / fibs[n]) * interval_len
    f_left = f(x_left)
    f_right = f(x_right)

    while n > 2:
        n -= 1
        if f_left < f_right:
            right_border = x_right
            x_right = x_left
            f_right = f_left
            interval_len = right_border - left_border
            x_left = left_border + (fibs[n - 2] / fibs[n]) * interval_len
            f_left = f(x_left)
        elif f_left > f_right:
            left_border = x_left
            x_left = x_right
            f_left = f_right
            interval_len = right_border - left_border
            x_right = left_border + (fibs[n - 1] / fibs[n]) * interval_len
            f_right = f(x_right)
        else:
            return (right_border + left_border) / 2, right_border - left_border
    return (right_border + left_border) / 2, right_border - left_border
