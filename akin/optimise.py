def hyperparameters(fixed_n, target_s, target_p):
    """ Optimise LSH hyperparameters for target Jaccard similarity and probability.

    Args:
        fixed_n
        target_s
        target_p

    """
    best_b, best_r = None, None
    min_error = float('inf')

    for r in range(1, fixed_n + 1):
        if fixed_n % r == 0:
            b = fixed_n // r
            p_actual = 1 - (1 - target_s ** r) ** b
            error = abs(p_actual - target_p)

            if error < min_error:
                min_error = error
                best_b, best_r = b, r

    actual_p = 1 - (1 - target_s ** best_r) ** best_b

    return best_b, best_r, actual_p
