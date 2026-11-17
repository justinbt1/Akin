def hyperparameters(fixed_n, target_s, target_p):
    """ Select LSH bands and rows closest to a target retrieval probability.

    Consider factor pairs where bands * rows equals fixed_n, minimising the
    absolute difference between target_p and 1 - (1 - target_s ** rows) ** bands.

    Args:
        fixed_n (int): Positive number of values in each MinHash signature.
        target_s (float): Target Jaccard similarity, between 0 and 1.
        target_p (float): Candidate retrieval probability at target_s, between 0 and 1.

    Returns:
        tuple[int, int, float]: Number of bands, number of rows per band, and etrieval
        probability at target_s.

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
