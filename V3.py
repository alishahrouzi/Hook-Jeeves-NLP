import numpy as np


def numerical_gradient(func, x, eps=1e-6):
    """Compute the numerical gradient of the function at point x."""
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_forward = np.copy(x)
        x_backward = np.copy(x)
        x_forward[i] += eps
        x_backward[i] -= eps
        grad[i] = (func(x_forward) - func(x_backward)) / (2 * eps)
    return grad


def armijo_condition(func, x, d, alpha, c):
    """Checks the Armijo condition."""
    grad = numerical_gradient(func, x)  # Compute gradient numerically
    return func(x + alpha * d) <= func(x) + c * alpha * np.dot(grad, d)


def golden_section_search(func, x, direction, alpha_low, alpha_high, tol=1e-4):
    """Golden section search to find optimal step size."""
    phi = (1 + np.sqrt(5)) / 2  
    resphi = 2 - phi
    
    alpha1 = alpha_low + resphi * (alpha_high - alpha_low)
    alpha2 = alpha_high - resphi * (alpha_high - alpha_low)
    
    while abs(alpha_high - alpha_low) > tol:
        f1 = func(x + alpha1 * direction)
        f2 = func(x + alpha2 * direction)
        
        if f1 < f2:
            alpha_high = alpha2
            alpha2 = alpha1
            alpha1 = alpha_low + resphi * (alpha_high - alpha_low)
        else:
            alpha_low = alpha1
            alpha1 = alpha2
            alpha2 = alpha_high - resphi * (alpha_high - alpha_low)
    
    return (alpha_low + alpha_high) / 2


def exploratory_search(func, x, alpha):
    """Exploratory search step."""
    n = len(x)
    new_x = np.copy(x).astype(float)  # Ensure float type for calculations
    
    for i in range(n):
        direction = np.zeros(n)
        direction[i] = 1  # Unit vector in the i-th direction
        
        if func(new_x + alpha * direction) < func(new_x):
            new_x += alpha * direction
        elif func(new_x - alpha * direction) < func(new_x):
            new_x -= alpha * direction
    
    return new_x


def hooke_jeeves(func, x0, alpha, tol=1e-4, max_iter=100, c=1e-4):
    """Hooke and Jeeves optimization algorithm."""
    x = np.copy(x0).astype(float)
    iter_count = 0
    
    while alpha > tol and iter_count < max_iter:

        new_x = exploratory_search(func, x, alpha)
        

        direction = new_x - x
        if np.linalg.norm(direction) < tol:  # اگر جهتی وجود نداشت
            break
        

        direction = direction / np.linalg.norm(direction)
        

        alpha_opt = golden_section_search(func, x, direction, 0, alpha)
        

        if armijo_condition(func, x, direction, alpha_opt, c):
            x = x + alpha_opt * direction
        else:
            alpha *= 0.5  
        
        iter_count += 1
    
    return x, func(x)


def sample_function(x):
    """Example function to minimize: f(x, y) = (x-3)^2 + (y-2)^2."""
    return (x[0] - 2)**4 + (x[0] - 2*x[1])**2


if __name__ == "__main__":
    x0 = np.array([0, 3])  
    alpha = 1.0  
    tol = 0.0023 
    max_iter = 10 
    
    optimal_x, optimal_value = hooke_jeeves(sample_function, x0, alpha, tol)
    
    print("Optimal Point:", optimal_x)
    print("Optimal Value:", optimal_value)
