import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, lambdify, sympify

x_sym, y_sym = symbols('x y')

def armijo_rule(f, grad, x, y, p, alpha=1.0, beta=0.5, sigma=0.1):
    """
    Implements the Armijo rule for step size adjustment.
    f: Objective function
    grad: Gradient of the function
    x, y: Current point
    p: Search direction
    alpha: Initial step size (default 1.0)
    beta: Reduction factor for alpha (default 0.5)
    sigma: Armijo condition constant (default 0.1)
    """
    while f(x + alpha * p[0], y + alpha * p[1]) > f(x, y) + sigma * alpha * np.dot(grad(x, y), p):
        alpha *= beta
    return alpha

def golden_section_search(f, x, y, p, a=0, b=1, tol=1e-5):
    """
    Golden Section Search for optimal step size.
    f: Objective function
    x, y: Current point
    p: Search direction
    a, b: Interval for step size
    tol: Tolerance for stopping condition
    """
    phi = (1 + np.sqrt(5)) / 2  
    resphi = 2 - phi

    c = a + resphi * (b - a)
    d = b - resphi * (b - a)

    while abs(b - a) > tol:
        if f(x + c * p[0], y + c * p[1]) < f(x + d * p[0], y + d * p[1]):
            b = d
        else:
            a = c

        c = a + resphi * (b - a)
        d = b - resphi * (b - a)

    return (a + b) / 2

def calculate_gradient(f):
    """
    Calculates the gradient of the objective function symbolically.
    f: Objective function in terms of x_sym and y_sym
    Returns: Callable gradient function
    """
    grad_x = diff(f, x_sym)
    grad_y = diff(f, y_sym)
    grad_func = lambdify((x_sym, y_sym), [grad_x, grad_y], 'numpy')
    return grad_func

def plot_optimization_path(points, f):
    """
    Plots the optimization path based on the sequence of points.
    points: List of points visited during optimization
    f: Objective function
    """
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    z_vals = [f(p[0], p[1]) for p in points]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, '-o', label='Optimization Path')
    plt.scatter(x_vals[-1], y_vals[-1], color='red', label='Optimal Point')


    x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f(xi, yi) for xi in x] for yi in y])
    plt.contour(X, Y, Z, levels=30, cmap='viridis')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimization Path')
    plt.legend()
    plt.colorbar(label='Objective Function Value')
    plt.show()

def hooke_jeeves_with_armijo_and_golden(f, grad, x0, tol):
    """
    Hooke and Jeeves optimization method with Armijo and Golden Section step size adjustment.
    f: Objective function
    grad: Gradient of the objective function
    x0: Initial point (tuple)
    tol: Tolerance for stopping condition
    """
    x, y = x0
    step_size = 1.0  
    base_point = np.array([x, y])
    iteration = 0

    points = [base_point]  

    while True:

        gradient_vector = np.array(grad(base_point[0], base_point[1]))
        direction = -gradient_vector  

   
        alpha_armijo = armijo_rule(f, grad, base_point[0], base_point[1], direction)
        alpha_golden = golden_section_search(f, base_point[0], base_point[1], direction)

       
        alpha = min(alpha_armijo, alpha_golden)

        trial_point = base_point + alpha * direction

    
        if np.linalg.norm(gradient_vector) < tol:
            break

     
        if f(trial_point[0], trial_point[1]) < f(base_point[0], base_point[1]):
            base_point = trial_point
            points.append(base_point)

        iteration += 1
        print("--------------------------------")
        print(f"Iteration {iteration}: \nPoint = {base_point}, \nFunction value = {f(base_point[0], base_point[1])}")

    plot_optimization_path(points, f)
    return base_point, f(base_point[0], base_point[1])


function_string = input("Enter the objective function (in terms of x and y): ")
objective_expr = sympify(function_string)  
objective_function = lambdify((x_sym, y_sym), objective_expr, 'numpy')


gradient_function = calculate_gradient(objective_expr)

x0 = tuple(map(float, input("Enter the initial point (x0, y0) separated by a comma: ").split(',')))
tol = float(input("Enter the tolerance: "))


optimal_point, optimal_value = hooke_jeeves_with_armijo_and_golden(objective_function, gradient_function, x0, tol)
print("--------------------------------")
print(f"Optimal point: {optimal_point}")
print(f"Optimal value: {optimal_value}")