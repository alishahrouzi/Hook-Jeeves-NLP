import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QLabel,
    QPushButton, QLineEdit, QWidget, QCheckBox
)
from sympy import symbols, sympify, lambdify


class HookeJeevesOptimizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hooke & Jeeves Optimizer")
        self.setGeometry(100, 100, 600, 750)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.function_input = QLineEdit()
        self.function_input.setPlaceholderText("Enter objective function (e.g., x**2 + y**2)")
        layout.addWidget(QLabel("Objective Function:"))
        layout.addWidget(self.function_input)

        self.initial_point_input = QLineEdit()
        self.initial_point_input.setPlaceholderText("Enter initial point (e.g., x=1, y=2)")
        layout.addWidget(QLabel("Initial Point:"))
        layout.addWidget(self.initial_point_input)

        self.step_size_input = QLineEdit()
        self.step_size_input.setPlaceholderText("Enter step size (e.g., 0.5, leave blank for auto)")
        layout.addWidget(QLabel("Step Size (leave blank for automatic calculation):"))
        layout.addWidget(self.step_size_input)

        self.auto_step_checkbox = QCheckBox("Use automatic step size calculation (Golden Section)")
        layout.addWidget(self.auto_step_checkbox)

        self.tolerance_input = QLineEdit()
        self.tolerance_input.setPlaceholderText("Enter tolerance (e.g., 1e-6)")
        layout.addWidget(QLabel("Tolerance:"))
        layout.addWidget(self.tolerance_input)

        self.run_button = QPushButton("Run Hooke & Jeeves")
        self.run_button.clicked.connect(self.run_hooke_jeeves)
        layout.addWidget(self.run_button)

        self.result_display = QLineEdit()
        self.result_display.setReadOnly(True)
        layout.addWidget(QLabel("Optimization Result:"))
        layout.addWidget(self.result_display)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    @staticmethod
    def parse_variables(input_text):
        try:
            variables = {}
            for pair in input_text.split(','):
                name, value = pair.strip().split('=')
                variables[name.strip()] = float(value.strip())
            return variables
        except Exception as e:
            raise ValueError(f"Error parsing variables: {e}. Use the format 'x=1, y=2'.")

    @staticmethod
    def evaluate_function(func, variables):
        try:
            sym_vars = {var: symbols(var) for var in variables}
            expr = sympify(func)
            func_numeric = lambdify(list(sym_vars.values()), expr, modules="numpy")
            return func_numeric(**variables)
        except Exception as e:
            raise ValueError(f"Error evaluating function: {e}")

    def armijo_condition(self, func, point, direction, step_size, sigma=1e-4):
        current_value = self.evaluate_function(func, point)
        new_point = {var: point[var] + direction[var] * step_size for var in point}
        new_value = self.evaluate_function(func, new_point)
        gradient_approx = (new_value - current_value) / step_size
        return new_value <= current_value + sigma * step_size * gradient_approx

    def golden_section_search_with_armijo(self, func, point, direction, a=0.0, b=1.0, tol=1e-6, sigma=1e-4):
        phi = (1 + np.sqrt(5)) / 2
        resphi = 2 - phi

        x1 = b - resphi * (b - a)
        x2 = a + resphi * (b - a)

        f1 = self.evaluate_function(func, {var: point[var] + direction[var] * x1 for var in point})
        f2 = self.evaluate_function(func, {var: point[var] + direction[var] * x2 for var in point})

        while abs(b - a) > tol:
            if not self.armijo_condition(func, point, direction, (a + b) / 2, sigma):
                b = (a + b) / 2
                continue

            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - resphi * (b - a)
                f1 = self.evaluate_function(func, {var: point[var] + direction[var] * x1 for var in point})
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + resphi * (b - a)
                f2 = self.evaluate_function(func, {var: point[var] + direction[var] * x2 for var in point})

        return (a + b) / 2

    def hooke_jeeves(self, func, initial_point, step_size, tolerance, auto_step):
        current_point = initial_point.copy()
        best_point = initial_point.copy()
        history = [initial_point.copy()]
        
        while step_size > tolerance:
            exploratory_result = self.exploratory_search(func, current_point, step_size)

            if self.evaluate_function(func, exploratory_result) <= self.evaluate_function(func, current_point):
                best_point = exploratory_result
                current_point = self.pattern_move(func, current_point, best_point)
                history.append(current_point.copy())
            else:
                if auto_step:
                    step_size = self.golden_section_search_with_armijo(func, current_point, {var: 1 for var in current_point}, a=0.01, b=step_size)
                else:
                    step_size *= 0.5
            print(best_point,step_size)
        return best_point, self.evaluate_function(func, best_point), history

    def exploratory_search(self, func, point, step_size):
        new_point = point.copy()
        for var in point:
            for direction in [1, -1]:
                test_point = new_point.copy()
                test_point[var] += direction * step_size
                if self.evaluate_function(func, test_point) < self.evaluate_function(func, new_point):
                    new_point = test_point
                    break
        return new_point

    def pattern_move(self, func, base_point, best_point):
        new_point = {
            var: best_point[var] + (best_point[var] - base_point[var])
            for var in base_point
        }
        return new_point if self.evaluate_function(func, new_point) < self.evaluate_function(func, best_point) else best_point

    def plot_results(self, func, history):
        if len(history[0]) != 2:
            return 

        x_vals = np.linspace(-10, 10, 500)
        y_vals = np.linspace(-10, 10, 500)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = self.evaluate_function(func, {"x": X, "y": Y})

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

    def run_hooke_jeeves(self):
        try:
            func = self.function_input.text()
            initial_point = self.parse_variables(self.initial_point_input.text())
            step_size = float(self.step_size_input.text() or 0.5)
            tolerance = float(self.tolerance_input.text())
            auto_step = self.auto_step_checkbox.isChecked()

            best_point, best_value, history = self.hooke_jeeves(func, initial_point, step_size, tolerance, auto_step)

            result_text = f"Optimal point: {best_point}\nOptimal value: {best_value}"
            self.result_display.setText(result_text)

            self.plot_results(func, history)

        except Exception as e:
            self.result_display.setText(f"Error: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HookeJeevesOptimizer()
    window.show()
    sys.exit(app.exec_())
