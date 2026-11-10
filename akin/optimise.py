def lsh_hyperparameters(fixed_n, target_s, target_p):
    best_b, best_r = None, None
    min_error = float('inf')
    
    print(f"Searching factors for fixed n = {fixed_n}...\n")
    
    for r in range(1, fixed_n + 1):
        if fixed_n % r == 0:
            b = fixed_n // r
            p_actual = 1 - (1 - target_s**r)**b
            error = abs(p_actual - target_p)
            
            if error < min_error:
                min_error = error
                best_b, best_r = b, r

    actual_p = 1 - (1 - target_s**best_r)**best_b
    
    print(f"🏆 Best Configuration:")
    print(f"  Bands (b): {best_b}")
    print(f"  Rows per band (r): {best_r}")
    print(f"  Actual Probability P({target_s}): {actual_p:.4f}")
    print(f"  Difference from target: {min_error:.4f}")


# Example usage:
# If you have exactly 200 minhashes, want a 0.7 similarity threshold, 
# and want a 95% (0.95) probability of catching them:
lsh_hyperparameters(fixed_n=200, target_s=0.7, target_p=0.95)
