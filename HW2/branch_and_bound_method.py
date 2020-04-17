from typing import Optional

import numpy as np

from HW2.simplex_method import simplex_method_noncanon, LinearProgrammingMethodResults

np.seterr(divide='ignore', invalid='ignore')


class BranchAndBoundMethodSolver:
    def __init__(self):
        self.max_bound = -1e30

    @staticmethod
    def is_integral(x) -> bool:
        return x - int(x) < 1e-30

    @staticmethod
    def _update_a(A, index, value) -> np.array:
        return np.vstack([A, [value if index == i else 0 for i in range(len(A[0]))]])

    @staticmethod
    def _update_b(b, value) -> np.array:
        return b + [value]

    def solve(self, A, b, c, search_min=True) -> Optional[LinearProgrammingMethodResults]:
        c = np.array(c, np.float)
        if search_min:
            c = -c
        self.max_bound = -1e30
        return self._solve_impl(A, b, c)

    def _solve_impl(self, A, b, c) -> Optional[LinearProgrammingMethodResults]:
        result = simplex_method_noncanon(A, b, c, False)
        if not result.success or result.get_value() < self.max_bound:
            return None
        for i, x_i in enumerate(result.x):
            if not BranchAndBoundMethodSolver.is_integral(x_i):
                integral_x_i = int(x_i)
                left_A = BranchAndBoundMethodSolver._update_a(A, i, 1)
                left_b = BranchAndBoundMethodSolver._update_b(b, integral_x_i)
                left_res = self._solve_impl(left_A, left_b, c)

                right_A = BranchAndBoundMethodSolver._update_a(A, i, -1)
                right_b = BranchAndBoundMethodSolver._update_b(b, -(integral_x_i + 1))
                right_res = self._solve_impl(right_A, right_b, c)
                if left_res is None:
                    return right_res
                if right_res is None:
                    return left_res
                return max([left_res, right_res], key=lambda x: x.value)
        self.max_bound = max(self.max_bound, int(result.get_value()))
        return result
