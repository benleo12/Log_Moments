import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Constants
alpha_s = 0.5
pi = np.pi
eps = 1e-8  # Small epsilon for numerical stability

# Define the prior distribution q(τ)
def q_tau(tau):
    return (2 * alpha_s / (3 * pi * (tau + eps))) * (-4 * np.log(tau + eps))
    #* np.exp(-2 * alpha_s * / (3 * pi) * (2 * np.log(tau + eps)**2))

def p_true_tau(tau):
    result = (2 * alpha_s / (3 * pi * tau)) * (-4 * np.log(tau )) * np.exp(- 2 * alpha_s / (3 * pi) * (2 * np.log(tau )**2))
    return result

# Lagrange multipliers
lambda_0 = torch.tensor(-1.0*0.5, requires_grad=True)
lambda_1 = torch.tensor(0.0, requires_grad=True)
lambda_2 = torch.tensor(4 * alpha_s / (3 * pi)*0.001, requires_grad=True)

# Optimizer (only optimizing the Lagrange multipliers now)
optimizer = optim.Adam([lambda_0 ,lambda_2], lr=0.01)

# Training loop
num_steps = 500000
num_samples = 10000

for step in range(num_steps):
    tau_samples = torch.rand(num_samples, 1)
    q_tau_values = torch.tensor(q_tau(tau_samples.numpy()), dtype=torch.float32)

    # Calculate p(tau) using the given formula
    p_tau_values = q_tau_values * torch.exp(-lambda_1 * torch.log(tau_samples + eps) - lambda_2 * torch.log(tau_samples + eps)**2)

    optimizer.zero_grad()

    # Loss calculation
    loss = torch.mean(p_tau_values * (torch.log(p_tau_values + eps) - torch.log(q_tau_values + eps)))

    # Constraints
    constraint_1 = torch.mean(p_tau_values) - 1
    constraint_2 = torch.mean(p_tau_values * torch.log(tau_samples + eps))  + (np.sqrt(3) * pi) / (4 * np.sqrt(alpha_s))
    constraint_3 = torch.mean(p_tau_values * torch.log(tau_samples + eps)**2) - (3 * pi) / (4 * alpha_s)

    # Combined loss with Lagrange multipliers
    total_loss =(lambda_2 * constraint_3)**2
    total_loss_2 = torch.abs(loss - lambda_0 * constraint_1 - lambda_2 * constraint_2)
    total_loss_3 = torch.abs(loss)
    total_loss.backward()
    optimizer.step()


    with torch.no_grad():
#     lambda_0.clamp_(1, 1)
#     lambda_1.clamp_( 4 * alpha_s / (3 * pi),  4 * alpha_s / (3 * pi))

     if step % 1000 == 0:
            print(f"Step: {step}, Loss: {total_loss.item()}, Lambda_0: {lambda_0.item()}, Lambda_1: {lambda_1.item()}, Lambda_2: {lambda_2.item()}")

            # Plotting the distributions
            tau_values = np.linspace(0.01, 1, 1000).reshape(-1, 1)
            tau_values_tensor = torch.tensor(tau_values, dtype=torch.float32)
            q_values = q_tau(tau_values)
            # Calculate p(τ) using the new formula
            p_values = q_values * np.exp(-lambda_1.item() * np.log(tau_values + eps)-lambda_2.item() * np.log(tau_values + eps)**2)

            # Evaluate p_true(τ) for the same tau values
            p_true_values = p_true_tau(tau_values)

            # Plotting the distributions
            plt.figure(figsize=(10, 5))
            plt.plot(tau_values, p_values/p_values.mean(), label=f'p(τ) at step {step}', color='blue')
            plt.plot(tau_values, p_true_values/p_true_values.mean(), label='p_true(τ)', color='green')
            plt.plot(tau_values, q_values/q_values.mean(), label='q(τ)', color='orange', linestyle='dashed')
            plt.title('p(τ) vs q(τ)')
            plt.xlabel('τ')
            plt.ylabel('Probability Density')
            plt.legend()
            plt.show()
