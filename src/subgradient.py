import numpy as np
from time import time

np.set_printoptions(precision=2, suppress=True)


def generic_subgradient(
    model,
    func_Z_lb,
    func_Z_ub,
    func_compute_subgradient,
    func_pi,
    u=[0, 0, 0],
    n_iter=10,
    early_stop=5,
    atol=1e-08,
    verbose=True,
    timeout=30 * 60,
):
    """
    Parameters:
        func_get_Z_lb: func
            function that receive the lagrange multiplier and return the lower bound
        func_compute_subgradient: func
            function that return the subgradients
        func_pi: func
            function that receive the iteration and return the current value of constant PI
        Z_ub: float
            model upper bound
        u: list or ndarray
            initial value for lagrange multipliers
        n_iter: int
            number of iterations
        atol: float
            The absolute tolerance parameter.
        verbose: bool
            if true print the iteration values
    Returns:
        status: int
            0 - Optimal value found
            1 - Maximum number of iterations reached
            2 - Non-improvement of Z_lb for early_stop iterarion
    """
    Z_lb_history = []
    Z_ub = np.inf
    start_time = time()

    # Termination criteria (1): number of iterations
    for k in range(1, n_iter + 1):
        Z_lb = func_Z_lb(u)
        Z_lb_history.append(Z_lb)

        # Termination criteria (3): Non-improvement of Z_lb
        es_check = Z_lb_history[-early_stop:]
        if len(es_check) == early_stop and all(
            [
                np.isclose(es_check[i - 1], es_check[i], atol=atol)
                for i in range(1, len(es_check))
            ]
        ):
            print(f"Non-improvement of Z_lb for {early_stop} iterations.")
            break

        # Termination criteria (4): optimality
        if np.isclose(Z_lb, Z_ub, atol=atol):
            print("Optimal value found.")
            break

        g = func_compute_subgradient()
        # Termination criteria (5): all subgradient is zero
        if all([np.isclose(g_i, 0, atol=atol) for g_i in g]):
            print("Optimal value found.")
            break

        Z_ub = min(Z_ub, func_Z_ub())
        pi = func_pi(k - 1)
        alpha = pi * ((Z_ub - Z_lb) / (g ** 2).sum())

        u = u + alpha * g
        u[u < 0] = 0

        if verbose:
            print("iteration:", k, ", Z_lb:", round(Z_lb, 2), ", Z_ub:", Z_ub)  # , ', u:', np.array(u))
        if time() - start_time >= timeout:
            print("Timedout reached.")
    if k >= n_iter:
        print("Maximum number of iterations reached.")
    return {"Z_lb": max(Z_lb_history), "Z_ub": Z_ub, "time": time() - start_time}


# Exemplo func_pi com valor fixo:
# func_pi = lambda k: 0.1
# Exemplo func_pi decrescendo em 1%
# func_pi = lambda k: (0.99**k) * 2