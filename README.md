# Hooke-Jeeves Optimization with Golden Section and Armijo Rule

## Introduction
This project implements the **Hooke-Jeeves** optimization method with step size adjustments using the **Golden Section Search** and **Armijo Rule**. The algorithm is designed for **nonlinear optimization problems**, where the objective function is given in terms of two variables (x, y). It allows users to input a function, an initial point, and a tolerance level to find the optimal point efficiently.

---
## What is Nonlinear Optimization?
Nonlinear optimization deals with the problem of finding the **minimum (or maximum) of a function** that is not necessarily linear. Unlike linear optimization, nonlinear problems may contain multiple local optima, requiring **gradient-based** or **direct search methods** to efficiently explore the solution space.

### Common Challenges in Nonlinear Optimization:
- Presence of multiple local minima or maxima
- Complex, non-convex objective functions
- Need for adaptive step size selection to avoid slow convergence or divergence

---
## Hooke-Jeeves Optimization Method
The **Hooke-Jeeves** method is a direct search optimization algorithm that iteratively improves a solution by exploring the function landscape. It consists of two main steps:
1. **Exploratory Move:** The algorithm evaluates function values at nearby points to find a better solution.
2. **Pattern Move:** If a better point is found, the search direction is reinforced in that direction to accelerate convergence.

The advantage of Hooke-Jeeves over purely gradient-based methods is that it can work even when derivatives are difficult to compute, making it useful for a variety of real-world applications.

---
## Step Size Adjustment Techniques
To improve convergence, the step size is dynamically controlled using two techniques:

### 1. Armijo Rule
The **Armijo rule** is a method for adaptive step size selection to ensure sufficient descent of the objective function. It iteratively reduces the step size **α** until the following condition is satisfied:
\[ f(x + \alpha p) \leq f(x) + \sigma \alpha \nabla f(x) \cdot p \]
where:
- \(\alpha\) is the step size
- \(\sigma\) is a small constant (typically 0.1)
- \(\nabla f(x)\) is the gradient
- \(p\) is the search direction

The method ensures that each step makes meaningful progress without excessively reducing function values.

### 2. Golden Section Search
The **Golden Section Search** is a technique for **finding an optimal step size** along a given direction. It works by iteratively reducing an interval **[a, b]** while maintaining the golden ratio **(≈ 1.618)** between test points. The method ensures fast convergence by **focusing only on promising regions**, making it highly effective for step size selection.

---
## Implementation Details
- Users **input an objective function** in terms of **x and y**.
- The **gradient of the function** is computed symbolically using **SymPy**.
- The **Hooke-Jeeves method** is applied with step size adjustments using **Armijo Rule** and **Golden Section Search**.
- The **optimization path** is plotted using **Matplotlib**, showing how the algorithm moves toward the optimal point.

---
## How to Use
1. **Run the script** and enter an objective function (e.g., `x**2 + y**2` for minimizing a quadratic function).
2. **Provide an initial point** (e.g., `0, 0`).
3. **Set a tolerance value** for convergence (e.g., `1e-5`).
4. The algorithm finds the optimal solution and **plots the optimization path**.

---
## Example Usage
```
Enter the objective function (in terms of x and y): x**2 + y**2
Enter the initial point (x0, y0) separated by a comma: 2, 3
Enter the tolerance: 1e-5
```
**Output:**
```
Iteration 1:
Point = [1.8, 2.7]
Function value = 10.53
...
Optimal point: [0, 0]
Optimal value: 0.0
```

---
## Dependencies
- Python 3.x
- `numpy`
- `matplotlib`
- `sympy`

To install required dependencies, run:
```bash
pip install numpy matplotlib sympy
```

---
## Conclusion
This project provides an **efficient and adaptive** implementation of the Hooke-Jeeves optimization method, enhanced with **Armijo Rule** and **Golden Section Search**. The combination of these techniques ensures reliable convergence in nonlinear optimization problems.

🚀 Happy optimizing!

