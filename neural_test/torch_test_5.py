import torch
import torch.optim as optim
import numpy as np

# Define the constants given in q(tau)
# Assuming alpha_s is a known constant, otherwise it needs to be defined or provided.
alpha_s = 0.3  # This is an example value, replace with the actual value.

# Define the integral equations as functions
def integral_equation_1(tau, lambda_0, lambda_1, lambda_2, alpha_s):
    q_tau = (2 * alpha_s / (3 * torch.pi)) * (-4 * torch.log(tau) / tau)
    integrand = q_tau * torch.exp(-1 - lambda_0 - lambda_1 * torch.log(tau)  - lambda_2 * torch.log(tau)**2)
    integral = torch.trapz(integrand, tau)
    return integral - 1

def integral_equation_2(tau, lambda_0, lambda_1, lambda_2, alpha_s):
    q_tau = (2 * alpha_s / (3 * torch.pi)) * (-4 * torch.log(tau) / tau)
    integrand = q_tau * torch.exp(-1 - lambda_0 - lambda_1 * torch.log(tau)  - lambda_2 * torch.log(tau)**2) * torch.log(tau)
    integral = torch.trapz(integrand, tau)
    return integral + np.sqrt(3) * torch.pi/(4 * np.sqrt(alpha_s))

def integral_equation_3(tau, lambda_0, lambda_1, lambda_2, alpha_s):
    q_tau = (2 * alpha_s / (3 * torch.pi)) * (-4 * torch.log(tau) / tau)
    integrand = q_tau * torch.exp(-1 - lambda_0 - lambda_1 * torch.log(tau)  - lambda_2 * torch.log(tau)**2) * torch.log(tau)**2
    integral = torch.trapz(integrand, tau)
    return integral - (3 * torch.pi) / (4 * alpha_s)

# Define the range of tau, ensuring to avoid tau = 0 to prevent division by zero and log(0)
tau = torch.linspace(1e-5, 1, steps=10000, requires_grad=False)

# Initialize the Lagrange multipliers with requires_grad=True so that PyTorch can compute gradients
lambda_0 = torch.tensor([-1*0.1], requires_grad=True)
lambda_1 = torch.tensor([(-8 * np.sqrt(alpha_s)/(torch.pi * np.sqrt(3)))*0.1], requires_grad=True)
lambda_2 = torch.tensor([4 * alpha_s / (3 * torch.pi)*0.1], requires_grad=True)

# Define the optimizer
optimizer = optim.Adam([lambda_0, lambda_1, lambda_2], lr=0.0001)

# Perform the optimization
for step in range(100000):
    optimizer.zero_grad()
    loss_1 = integral_equation_1(tau, lambda_0, lambda_1, lambda_2, alpha_s)**2
    loss_2 = integral_equation_2(tau, lambda_0, lambda_1, lambda_2, alpha_s)**2
    loss_3 = integral_equation_3(tau, lambda_0, lambda_1, lambda_2, alpha_s)**2
    loss = loss_1 + loss_2 + loss_3  # Total loss is the sum of the losses from both equations
    loss.backward()
    optimizer.step()

    # Print the progress every 500 steps
    if step % 5000 == 0:
        print(f"Step {step}, Loss: {loss.item()}, Lambda 0: {lambda_0.item()}, Lambda 1: {lambda_1.item()}, , Lambda 2: {lambda_2.item()}")

# Print final values
print(f"Final Lambda 0: {lambda_0.item()}, Final Lambda 1: {lambda_1.item()}, , Final Lambda 2: {lambda_2.item()}")
