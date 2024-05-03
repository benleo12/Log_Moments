import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad

tau = np.linspace(1e-5, 1, 1000)

# Define the constants given in q(tau)
alpha_s = 0.3  # This is an example value, replace with the actual value.

# Define the function q(tau)
def q(tau, alpha_s):
    return (2 * alpha_s / (3 * np.pi)) * (-4 * np.log(tau) / tau)

# Define the integral equations
def integral_equation_1(lambda_0, lambda_1, alpha_s, tau):
    integrand = lambda t: q(t, alpha_s) * np.exp(-1 - lambda_0 - lambda_1 * np.log(t)**2)
    result, _ = quad(integrand, 0, np.inf)
    return result - 1

def integral_equation_2(lambda_0, lambda_1, alpha_s, tau):
    integrand = lambda t: q(t, alpha_s) * np.exp(-1 - lambda_0 - lambda_1 * np.log(t)**2) * np.log(t)**2
    result, _ = quad(integrand, 0, np.inf)
    return result - (3 * np.pi) / (4 * alpha_s)

# Define the system of equations
def equations(p):
    lambda_0, lambda_1 = p
    return (integral_equation_1(lambda_0, lambda_1, alpha_s, tau),
            integral_equation_2(lambda_0, lambda_1, alpha_s, tau))

# Initial guess for the Lagrange multipliers
initial_guess = (-1, 4 * alpha_s / (3 * np.pi))

# Solve the system of equations
solution = fsolve(equations, initial_guess)

print(f"Numerical solution: Lambda_0 = {solution[0]}, Lambda_1 = {solution[1]}")

