import numpy as np
np.set_printoptions(precision=2, suppress= True)


def generic_subgradient(func_get_Z_lb, func_compute_subgradient, func_pi, Z_ub, u=[0, 0, 0], n_iter=10, early_stop=5, atol=1e-08, verbose=True):
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
    # Termination criteria (1): number of iterations
    for k in range(1, n_iter+1):
        Z_lb = func_get_Z_lb(u)
        Z_lb_history.append(Z_lb)

        # Termination criteria (3): Non-improvement of Z_lb
        es_check = Z_lb_history[-early_stop:]
        if len(es_check) == early_stop and all([np.isclose(es_check[i-1], es_check[i], atol=atol) for i in range(1, len(es_check))]):
            print('Stopping because of Non-improvement of Z_lb.')
            return 2
        
        # Termination criteria (4): optimality
        if np.isclose(Z_lb, Z_ub, atol=atol):
            print('Optimal value found.')
            return 0

        g = func_compute_subgradient()
        # Termination criteria (5): all subgradient is zero
        if all([np.isclose(g_i, 0, atol=atol) for g_i in g]):
            print('Optimal value found.')
            return 0

        alpha = func_pi(k - 1) * ((Z_ub - Z_lb) / (g ** 2).sum())

        u = u + alpha * g
        u[u < 0] = 0
        
        if verbose:
            print('iteration:', k, ', Z_lb:', round(Z_lb, 2), ', u:', np.array(u))
   
    print('Stopping because of Maximum number of iterations reached.')
    return 1
# Exemplo func_pi com valor fixo:
# func_pi = lambda k: 0.1
# Exemplo func_pi decrescendo em 1%
# func_pi = lambda k: (0.99**k) * 2