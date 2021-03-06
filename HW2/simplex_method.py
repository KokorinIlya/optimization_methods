from typing import Optional

import numpy as np

np.seterr(divide='ignore', invalid='ignore')


class LinearProgrammingMethodResults:
    def __init__(self, value, x, search_min):
        self._value = value
        self.x = x
        self.search_min = search_min

    def get_value(self):
        return self._value if self.search_min else -self._value

    def __str__(self):
        return 'fun: {}\n' \
               'x: {}\n'.format(self.get_value(), self.x)


def simplex_method_noncanon(a, b, c, search_min=True) -> Optional[LinearProgrammingMethodResults]:
    n, m = np.shape(a)

    # добавим единичную матрицу чтобы перевести неравенства в равенства
    a = np.hstack((np.array(a, np.float), np.eye(n)))

    # новые переменные будут входить в функцию с коэффициентом 0
    c = np.hstack((np.array(c, np.float), np.zeros(n, np.float)))

    res = simplex_method_canon(a, b, c, search_min)
    if res is None:
        return None
    res.x = res.x[:m]
    return res


def simplex_method_canon(A, b, c, search_min=True) -> Optional[LinearProgrammingMethodResults]:
    n, m = np.shape(A)

    c = np.array(c, np.float)
    if search_min:
        c = -c
    M = 1e10

    # добавим единичную матрицу для искуственного базиса
    A = np.hstack((np.array(A, np.float), np.eye(len(A))))

    # добавим штраф для искусственного базиса
    c = np.hstack((c, np.repeat(-M, n)))

    # приведем систему уравнений к виду ax=b, где все b>=0
    for i in range(n):
        if b[i] != 0:
            A[i] *= np.sign(b[i])
            b[i] *= np.sign(b[i])
    b = np.array(b, np.float)

    # избавляемся от коэффициентов M чтобы Q(x, x_new) = 0
    q0 = 0
    for i in range(n):
        c = c + A[i] * M
        q0 += b[i] * M

    # начальное решение X = 0, X_new (базис) = b
    basis = np.arange(n) + m
    eps = 1e-10

    while True:
        # находим разрешающий элемент (можно не максимум, достаточно положительный)
        l = np.argmax(c)
        # если положительных нет, то оптимизация закончена
        if c[l] <= eps:
            break

        # выбираем разрешающую строку r
        resolving_col = np.divide(b, A[:, l])
        cur_min = np.inf
        r = -1
        for i in range(n):
            if A[i][l] > eps and resolving_col[i] <= cur_min:
                r = i
                cur_min = resolving_col[i]
        if r == -1:
            return None

        # нашли разрешаюший элемент
        resolving_el = A[r][l]

        # вносим новую переменную в базис
        basis[r] = l

        # Шаг модифицированного жорданова исключения
        for i in range(n):
            if i == r:
                mul = c[l] / resolving_el
                c -= A[r] * mul
                q0 -= b[r] * mul
            else:
                mul = A[i, l] / resolving_el
                A[i] -= A[r] * mul
                b[i] -= b[r] * mul

    # если значение функции оказалось больше M, то метод разошелся
    if abs(q0) > M:
        return None

    x = np.zeros(m)
    for i in range(n):
        r = basis[i]
        # если добавочная переменная вошла в базис и не зануляется, то решения не существует
        if r >= m and abs(b[i] / A[i][r]) > eps:
            return None
        else:
            # если переменная не добавочная, то просто расчитываем ее
            x[r] = b[i] / A[i][r]
    return LinearProgrammingMethodResults(q0, x, search_min)
