import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Constants
alpha_s = 0.5
pi = np.pi
eps = 0.0000001

# Define the prior distribution q(τ)
def q_tau(tau):
    return (2 * 1.001*alpha_s / (3 * pi * (tau + eps))) * (-4 * np.log(tau + eps ))* np.exp(- 2 * alpha_s*1.001 / (3 * pi) * (2 * np.log(tau + eps)**2))
#* np.exp(- 2 * alpha_s / (3 * pi) * (2 * np.log(tau )**2))
#* np.exp(-alpha_s* np.log(tau + eps)**2)

def p_true_tau(tau):
    result = (2 * alpha_s / (3 * pi * tau)) * (-4 * np.log(tau )) * np.exp(- 2 * alpha_s / (3 * pi) * (2 * np.log(tau )**2))
    return result

# Define the Neural Network
class ProbabilityDensityNetwork(nn.Module):
    def __init__(self):
        super(ProbabilityDensityNetwork, self).__init__()
        self.dense_1 = nn.Linear(1, 256)
        self.dense_2 = nn.Linear(256, 256)
        self.dense_3 = nn.Linear(256, 128)
        self.dense_4 = nn.Linear(128, 128)
        self.dense_5 = nn.Linear(128, 64)
        self.dense_6 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 1)
        self.activation = nn.Softplus()

    def forward(self, x):
        x = self.activation(self.dense_1(x))
        x = self.activation(self.dense_2(x))
        x = self.activation(self.dense_3(x))
        x = self.activation(self.dense_4(x))
        x = self.activation(self.dense_5(x))
        x = self.activation(self.dense_6(x))
        return torch.exp(self.output_layer(x)) # Using exp to ensure positive output

# Instantiate the model
model = ProbabilityDensityNetwork()

# Lagrange multipliers
lambda_0 = torch.tensor(-1.0 , requires_grad=True)
lambda_1 = torch.tensor(-np.sqrt(alpha_s) * 4 / (np.sqrt(3) * pi) , requires_grad=True)
lambda_2 = torch.tensor( 4 * alpha_s / (3 * pi), requires_grad=True)

# Optimizer
optimizer = optim.Adam(list(model.parameters()) + [lambda_0, lambda_1, lambda_2], lr=0.001)

# Training the model
num_steps = 50000
num_samples = 1000

def sigma(x_values, p_values, x_i):
    mask = x_values > x_i
    sum_greater_x_i = torch.sum(p_values[mask])
    total_sum = torch.sum(p_values) + eps
    return sum_greater_x_i / total_sum

def constraint_1_individual(p_tau_i):
    return p_tau_i - 1

def constraint_2_individual(p_tau_i, tau_i):
    term1 = p_tau_i * torch.log(tau_i+eps)
    term2 = (np.sqrt(3) * np.pi) / (4 * np.sqrt(alpha_s))
    return term1 + term2

def constraint_3_individual(p_tau_i, tau_i):
    term1 = p_tau_i * torch.log(tau_i+eps)**2
    term2 = (3 * np.pi) / (4 * alpha_s)
    return term1 - term2

# Assuming model, constraints, optimizer, num_steps, num_samples, and q_tau are defined
for step in range(num_steps):
    tau_samples = torch.rand(num_samples, 1)
    q_tau_values = torch.tensor(q_tau(tau_samples.numpy()), dtype=torch.float32)

    optimizer.zero_grad()
    p_tau_values = model(tau_samples)

    loss = 0
    for i in range(num_samples):
        x_i = tau_samples[i]
        sigma_p = sigma(tau_samples, p_tau_values, x_i)
        sigma_q = sigma(tau_samples, q_tau_values, x_i)
        sigma_p = torch.clamp(sigma_p, min=eps)
        sigma_q = torch.clamp(sigma_q, min=eps)
        p_tau_i = p_tau_values[i]
        loss = loss + (sigma_p - sigma_q)**2 #- lambda_0*constraint_1_individual(p_tau_i) - lambda_1*constraint_3_individual(p_tau_i,x_i)

    loss = loss / num_samples  # Average the loss over the batch
    total_loss = loss
    loss.backward()
    optimizer.step()




    with torch.no_grad():
#     lambda_0.clamp_(1, 1)
#     lambda_1.clamp_( 4 * alpha_s / (3 * pi),  4 * alpha_s / (3 * pi))

     if step % 1000 == 0:
      print(f"Step: {step}, Loss: {total_loss.item()}, Lambda_0: {lambda_0.item()}, Lambda_1: {lambda_1.item()}", Lambda_2: {lambda_2.item()}")

      # Plotting the distributions
      tau_values = np.linspace(0.01, 1, 1000)
      p_values = model(torch.tensor(tau_values.reshape(-1, 1), dtype=torch.float32)).detach().numpy().flatten()
      q_values = q_tau(tau_values)
      # Evaluate p_true(τ) for the same tau values using the PyTorch function
      p_true_values = p_true_tau(tau_values)

      # Plotting the distributions
      plt.figure(figsize=(10, 5))
      plt.plot(tau_values, p_values/p_values.mean(), label=f'p(τ) at step {step}', color='blue')
      plt.plot(tau_values, p_true_values/p_true_values.mean(), label=f'p_true(τ)', color='green')
      plt.plot(tau_values, q_values/q_values.mean(), label='q(τ)', color='orange', linestyle='dashed')
      plt.title('p(τ) vs q(τ)')
      plt.xlabel('τ')
      plt.ylabel('Probability Density')
      plt.legend()
      plt.show()
