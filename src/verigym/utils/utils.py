import z3
from typing import Callable

def identity_map(x):
    return x

def check_sat_label(lb, ub, label: Callable, check_not: bool):
    """
    Checks whether the function in label is satisfied/violated.
    Used for checking state labels for over/underapproximation on abstractions from continuous states.
    
    Parameters
    ----------
    lb : np.array
        lower bound values per dimension
    ub : np.array
        upper bound values per dimension
    label : Callable
    check_not : bool
        if True, we want to check if (not label == True) somewhere. 
        If false, we want to check if (label == True) somewhere.

    Returns
    -------
    check : bool, True if sat, False if unsat

    Example
    -------
    ```
    # box bounds on a state, like:
    box_bounds_2d = [[-1, 1], [-2, 2]]
    # we pass to this function lower and upper bounds:
    lb = np.array([-1, -2])
    ub = np.array([1, 2])
    ```
    """
    dims = len(lb)
    X = [z3.Real(f'x{i}') for i in range(dims)]
    s = z3.Solver()
    for x, lower, upper in zip(X, lb, ub):
        s.add(x >= lower, x <= upper)
    if check_not:
        s.add(z3.Not(label(X)))
    else:
        s.add(label(X))

    return s.check() == z3.sat