import torch
import torch.optim as optim
import numpy as np

# Assuming alpha_s is a known constant
alpha_s = 0.3
steps=10000

# Dataset of {tau_i, q_tau_i} pairs
tau_i = torch.linspace(1e-5, 1, steps)
x_i = torch.rand(steps)
tau_i = 1e-5**(torch.sqrt(x_i))

q_tau_i = 1

# Manual integration function using Riemann sum
def manual_integration(integrand, tau_values):
    C= 4*2*alpha_s/(3*np.pi)
    integral = torch.mean(integrand)*C/2*np.log(1e-5)**2
    return integral

# Define the integral equations using the dataset directly
def integral_equation_1_direct(lambda_0, lambda_1, lambda_2):
    integrand = torch.exp(-1 - lambda_0 - lambda_1 * torch.log(tau_i) - lambda_2 * torch.log(tau_i)**2)
    integral = manual_integration(integrand, tau_i)
    return integral - 1

def integral_equation_2_direct(lambda_0, lambda_1, lambda_2):
    integrand = torch.exp(-1 - lambda_0 - lambda_1 * torch.log(tau_i) - lambda_2 * torch.log(tau_i)**2) * torch.log(tau_i)
    integral = manual_integration(integrand, tau_i)
    return integral + np.sqrt(3) * torch.pi / (4 * np.sqrt(alpha_s))

def integral_equation_3_direct(lambda_0, lambda_1, lambda_2):
    integrand = torch.exp(-1 - lambda_0 - lambda_1 * torch.log(tau_i) - lambda_2 * torch.log(tau_i)**2) * torch.log(tau_i)**2
    integral = manual_integration(integrand, tau_i)
    return integral - (3 * torch.pi) / (4 * alpha_s)

# Initialize the Lagrange multipliers
lambda_0 = torch.tensor([-1*0.1], requires_grad=True)
lambda_1 = torch.tensor([(-8 * np.sqrt(alpha_s)/(torch.pi * np.sqrt(3)))*0.1], requires_grad=True)
lambda_2 = torch.tensor([4 * alpha_s / (3 * torch.pi)*0.1], requires_grad=True)

# Define the optimizer
optimizer = optim.Adam([lambda_0, lambda_1, lambda_2], lr=0.001)

# Optimization loop
for step in range(100000):
    optimizer.zero_grad()
    loss_1 = integral_equation_1_direct(lambda_0, lambda_1, lambda_2)**2
    loss_2 = integral_equation_2_direct(lambda_0, lambda_1, lambda_2)**2
    loss_3 = integral_equation_3_direct(lambda_0, lambda_1, lambda_2)**2
    loss = loss_1 + loss_2 + loss_3  # Total loss
    loss.backward()
    optimizer.step()

    if step % 5000 == 0:
        print(f"Step {step}, Loss: {loss.item()}, Lambda 0: {lambda_0.item()}, Lambda 1: {lambda_1.item()}, Lambda 2: {lambda_2.item()}")

# Print final values of the Lagrange multipliers
print(f"Final Lambda 0: {lambda_0.item()}, Final Lambda 1: {lambda_1.item()}, Final Lambda 2: {lambda_2.item()}")
