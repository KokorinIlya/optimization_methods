import numpy


# TODO: scipy.linalg.cho_factor, scipy.linalg.cho_solve
def newton(f, f_grad, f_hess, start_arg, stop_criterion, eps=1e-5, max_iters=100):
    assert stop_criterion in {'arg', 'value', 'delta'}
    cur_arg = start_arg
    cur_value = f(cur_arg)
    trace = [cur_arg]
    while True:
        cur_grad = f_grad(cur_arg)
        cur_hess = f_hess(cur_arg)
        cur_delta = numpy.matmul(cur_grad, numpy.linalg.inv(cur_hess))
        next_arg = cur_arg - cur_delta
        next_value = f(next_arg)
        trace.append(next_arg)

        if len(trace) == max_iters:
            raise ArithmeticError()

        if (stop_criterion == 'arg' and numpy.linalg.norm(next_arg - cur_arg) < eps) or \
                (stop_criterion == 'value' and abs(next_value - cur_value) < eps) or \
                (stop_criterion == 'delta' and numpy.linalg.norm(cur_delta) < eps):
            return trace
        cur_arg = next_arg
        cur_value = next_value
