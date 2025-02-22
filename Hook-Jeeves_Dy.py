import numpy as np
from sympy import symbols, sympify, lambdify
import matplotlib.pyplot as plt

def parse_variables(input_text):
    try:
        variables = {}
        for pair in input_text.split(','):
            name, value = pair.strip().split('=')
            variables[name.strip()] = float(value.strip())
        return variables
    except Exception as e:
        raise ValueError(f"Error parsing variables: {e}. Use the format 'x=1, y=2'.")

def evaluate_function(func, variables):
    try:
        sym_vars = {var: symbols(var) for var in variables}
        expr = sympify(func)
        func_numeric = lambdify(list(sym_vars.values()), expr, modules="numpy")
        return func_numeric(**variables)
    except Exception as e:
        raise ValueError(f"Error evaluating function: {e}")

def golden_section_search(func, point, direction, a=0, b=1, tol=1e-6):
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi

    x1 = b - resphi * (b - a)
    x2 = a + resphi * (b - a)

    f1 = evaluate_function(func, {var: point[var] + direction[var] * x1 for var in point})
    f2 = evaluate_function(func, {var: point[var] + direction[var] * x2 for var in point})

    while abs(b - a) > tol:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - resphi * (b - a)
            f1 = evaluate_function(func, {var: point[var] + direction[var] * x1 for var in point})
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + resphi * (b - a)
            f2 = evaluate_function(func, {var: point[var] + direction[var] * x2 for var in point})
    step_size = (a + b) / 2
    # print(f"Golden Section Search - Step Size: {step_size}")
    return step_size

def exploratory_search(func, point):
    new_point = point.copy()
    for var in point:
        direction = {v: 0 for v in point}
        direction[var] = 1
        step_size = golden_section_search(func, new_point, direction)
        test_point = {v: new_point[v] + step_size * direction[v] for v in new_point}

        if evaluate_function(func, test_point) < evaluate_function(func, new_point):
            new_point = test_point
        else:
            direction[var] = -1
            step_size = golden_section_search(func, new_point, direction)
            test_point = {v: new_point[v] + step_size * direction[v] for v in new_point}
            if evaluate_function(func, test_point) < evaluate_function(func, new_point):
                new_point = test_point

    return new_point

def pattern_move(func, base_point, best_point):
    direction = {var: best_point[var] - base_point[var] for var in base_point}
    step_size = golden_section_search(func, best_point, direction)
    new_point = {var: best_point[var] + step_size * direction[var] for var in base_point}

    return new_point if evaluate_function(func, new_point) < evaluate_function(func, best_point) else best_point

def hooke_jeeves(func, initial_point, tolerance):
    current_point = initial_point.copy()
    best_point = initial_point.copy()
    history = [initial_point.copy()]
    iteration_info = []

    iteration = 0
    while True:
        iteration += 1
        exploratory_result = exploratory_search(func, current_point)

        if evaluate_function(func, exploratory_result) < evaluate_function(func, current_point):
            best_point = exploratory_result
            current_point = pattern_move(func, current_point, best_point)
            history.append(current_point.copy())
            iteration_info.append({
                "iteration": iteration,
                "current_point": current_point.copy(),
                "current_value": evaluate_function(func, current_point)
            })
        else:
            break

    return best_point, evaluate_function(func, best_point), history, iteration_info

def plot_results(func, history):
    if len(history[0]) != 2:
        return 

    x_vals = np.linspace(-10, 10, 500)
    y_vals = np.linspace(-10, 10, 500)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = evaluate_function(func, {"x": X, "y": Y})

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar()

    history_x = [point['x'] for point in history]
    history_y = [point['y'] for point in history]
    plt.plot(history_x, history_y, 'r-o', label="Optimization Path")

    plt.title("Optimization Path")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def main():
    try:
        func = input("Enter objective function (e.g., x**2 + y**2): ")
        initial_point_input = input("Enter initial point (e.g., x=1, y=2): ")
        initial_point = parse_variables(initial_point_input)
        tolerance = float(input("Enter tolerance (e.g., 1e-6): "))

        best_point, best_value, history, iteration_info = hooke_jeeves(func, initial_point, tolerance)
        
        print("\nOptimization Completed")
        print(f"Optimal point: {best_point}")
        print(f"Optimal value: {best_value}")
        print(f"Total Iterations: {len(iteration_info)}")

        for info in iteration_info:
            print(f"Iteration {info['iteration']}: Point={info['current_point']}, Value={info['current_value']}")

        plot_results(func, history)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
