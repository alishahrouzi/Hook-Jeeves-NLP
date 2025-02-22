import numpy as np

# Define the objective function (Rosenbrock function)
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# Analytical gradient of the Rosenbrock function
def rosenbrock_grad(x):
    dfdx0 = -400 * (x[1] - x[0]**2) * x[0] - 2 * (1 - x[0])
    dfdx1 = 200 * (x[1] - x[0]**2)
    return np.array([dfdx0, dfdx1])

# Golden section search for line minimization
def golden_section_search(f, a, b, tol=1e-5):
    inv_phi = (np.sqrt(5) - 1) / 2  # Golden ratio conjugate
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    while abs(b - a) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - inv_phi * (b - a)
        d = a + inv_phi * (b - a)
    return (a + b) / 2

# Armijo's condition implementation
def armijo_condition(f, x, p, alpha, c1=1e-4, max_iter=20):
    f_x = f(x)
    grad = rosenbrock_grad(x)
    directional_derivative = np.dot(grad, p)
    if directional_derivative >= 0:
        raise ValueError("Search direction is not a descent direction.")
    for _ in range(max_iter):
        if f(x + alpha * p) <= f_x + c1 * alpha * directional_derivative:
            return alpha
        alpha /= 2
    return alpha

# Hooke and Jeeves algorithm with combined line search
def hooke_jeeves(f, x0, step_sizes, max_iterations=1000, tol=1e-6):
    x = np.array(x0, dtype=float)
    n = len(x)
    best = x.copy()
    f_best = f(x)
    iteration = 0
    while iteration < max_iterations:
        x_trial = best.copy()
        grad = rosenbrock_grad(best)
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1
            # Ensure the search direction is a descent direction
            if grad[i] > 0:
                p = -e_i  # Move in the negative direction
            else:
                p = e_i   # Move in the positive direction
            # Define the line search function
            def line_func(alpha):
                return f(best + alpha * p)
            # Use golden section search to find the optimal step size
            alpha_opt = golden_section_search(line_func, 0, step_sizes[i])
            # Apply Armijo's condition to adjust step size
            alpha = armijo_condition(f, best, p, alpha_opt)
            x_trial[i] = best[i] + alpha * p[i]
            if f(x_trial) < f(best):
                best = x_trial.copy()
                f_best = f(best)
            else:
                x_trial[i] = best[i]
        # Check for convergence
        if np.linalg.norm(x - best) < tol:
            break
        x = best.copy()
        iteration += 1
    return best, f_best, iteration

# Main function to run the algorithm
def main():
    # Initial guess
    x0 = np.array([-1.2, 1.0])
    # Initial step sizes
    step_sizes = np.array([0.5, 0.5])
    # Run Hooke and Jeeves algorithm
    x_opt, f_opt, iter_num = hooke_jeeves(rosenbrock, x0, step_sizes)
    print("Optimal solution:", x_opt)
    print("Optimal function value:", f_opt)
    print("Number of iterations:", iter_num)

if __name__ == "__main__":
    main()