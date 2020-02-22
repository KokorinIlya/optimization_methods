import numpy.linalg
from HW1.one_demensional import line_search


def gradient_descent(f, f_grad, start_arg, step_chooser, stop_criterion, eps=1e-5):
    cur_arg = start_arg
    cur_value = f(cur_arg)
    trace = [cur_arg]
    while True:
        cur_grad = f_grad(cur_arg)
        cur_step = step_chooser(f, cur_grad, cur_arg)
        next_arg = cur_arg - cur_step * cur_grad
        next_value = f(next_arg)
        trace.append(next_arg)
        if (stop_criterion == 'arg' and numpy.linalg.norm(next_arg - cur_arg) < eps) or \
                (stop_criterion == 'value' and abs(next_value - cur_value) < eps) or \
                (stop_criterion == 'grad' and numpy.linalg.norm(cur_grad) < eps):
            return next_arg, trace
        cur_arg = next_arg
        cur_value = next_value


def linear_step_chooser(method):
    def result(f, grad, arg):
        def linear_optimization_problem(k):
            return f(arg - k * grad)

        left_border = 0.
        right_border = line_search(linear_optimization_problem, left_border)
        answer, _ = method(linear_optimization_problem, left_border, right_border)
        return answer

    return result
