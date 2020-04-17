import numpy as np

from HW2.branch_and_bound_method import BranchAndBoundMethodSolver
from HW2.simplex_method import simplex_method_canon, simplex_method_noncanon


def print_simplex(A, b, c, search_min):
    print("Simplex method canonical")
    print(simplex_method_canon(A, b, c, search_min))
    print("Simplex method non-canonical")
    print(simplex_method_noncanon(A, b, c, search_min))


def print_branch_and_bound(A, b, c, search_min):
    solver = BranchAndBoundMethodSolver()
    print("Branch and Bound method")
    print(solver.solve(A, b, c, search_min))


def print_variant(A, b, c, search_min=True):
    print("#" * 40)
    print(f"A\n{A}")
    print(f"b\n{b}")
    print(f"c\n{c}")
    print_simplex(A, b, c, search_min)
    print_branch_and_bound(A, b, c, search_min)


A = [[1, 0, 0, 1, -2],
     [0, 1, 0, -2, 1],
     [0, 0, 1, 3, 1]]
b = [1, 2, 3]
c = [0, 0, 0, 1, -1]

A = -np.array(A)
b = -np.array(b)

print_variant(A, b, c)

A = [[-1, 1, 1, 0],
     [1, -2, 0, 1]]
b = [1, 2]
c = [-1, -1, 0, 0]

A = -np.array(A)
b = -np.array(b)

print_variant(A, b, c)

A = [[2, 1, 0],
     [1, 2, -2],
     [0, 1, 2]]

b = [10, 20, 5]
c = [2, -1, 2]

print_variant(A, b, c, False)

A = [[2, 3, 1, 2, 1],
     [2, 1, -3, 2, 1],
     [2, 1, 2, 1, 0]]

b = [1, 3, 1]
c = [-1, 1, -2, 1, 5]

print_variant(A, b, c)

A = [[4, 3],
     [-4, 3],
     [1, 0],
     [0, -1]]

b = [22, 2, 2, -4]
c = [-5, -4]

print_variant(A, b, c, False)

# Example from book

A = [[4, 3],
     [-4, 3]]
b = [22, 2]
c = [-5, 4]
print_variant(A, b, c, False)
