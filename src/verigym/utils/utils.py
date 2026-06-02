import z3

def identity_map(x):
    return x

def check_sat_label(lb, ub, label, check_not):
    """
    Checks whether the function in label is satisfied/violated.
    Used for checking state labels for over/underapproximation on abstractions from continuous states.
    
    Parameters
    ----------
    lb : np.array, lower bound values per dimension
    ub : np.array, upper bound values per dimension
    label : function over the dimensions
    check_not : bool

    Returns
    -------
    check : bool, True if sat, False if unsat
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