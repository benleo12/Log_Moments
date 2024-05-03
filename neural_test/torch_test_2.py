import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Constants
alpha_s = 0.5
pi = np.pi
eps = 0.0 #0.0000001

# Define the prior distribution q(τ)
def q_tau(tau):
    return (2.5 * alpha_s / (3 * pi * (tau + eps))) * (-4 * np.log(tau + eps ))* np.exp(- 2.5 * alpha_s / (3 * pi) * (2 * np.log(tau + eps)**2))
#* np.exp(- 2.5 * alpha_s / (3 * pi) * (2 * np.log(tau )**2))
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
lambda_0 = torch.tensor(-1.0*0.5 , requires_grad=True)
lambda_1 = torch.tensor(-np.sqrt(alpha_s) * 4 / (np.sqrt(3) * pi)*0.0 , requires_grad=True)
lambda_2 = torch.tensor( 4 * alpha_s / (3 * pi)*1.5, requires_grad=True)

# Optimizer
optimizer = optim.Adam(list(model.parameters()) + [lambda_0, lambda_2], lr=0.001)

# Training the model
num_steps = 50000
num_samples = 1000

for step in range(num_steps):
    tau_samples = torch.rand(num_samples, 1)
    q_tau_values = torch.tensor(q_tau(tau_samples.numpy()), dtype=torch.float32)

    optimizer.zero_grad()
    p_tau_values = model(tau_samples)*q_tau_values
    p_true_tau_values = torch.tensor(p_true_tau(tau_samples.numpy()), dtype=torch.float32)
    loss = torch.mean(p_tau_values*(torch.log(p_tau_values) - torch.log(p_true_tau_values)))
    # Constraints
    constraint_1 = torch.mean(p_tau_values) - 1
    constraint_2 = torch.mean(p_tau_values * torch.log(tau_samples))/torch.mean(p_tau_values) + (np.sqrt(3) * pi)/(4 * np.sqrt(alpha_s))
    constraint_3 = torch.mean(p_tau_values * torch.log(tau_samples)**2)/torch.mean(p_tau_values) - (3 * pi)/(4 * alpha_s)
    # Combined loss with Lagrange multipliers
    total_loss =  loss + torch.abs(lambda_2 * constraint_3)
    total_loss_2 = loss - lambda_0*constraint_1 - lambda_2 * constraint_2
    total_loss_3 = loss
    total_loss = loss/num_samples
    total_loss.backward()
    optimizer.step()


    with torch.no_grad():
#     lambda_0.clamp_(1, 1)
#     lambda_1.clamp_( 4 * alpha_s / (3 * pi),  4 * alpha_s / (3 * pi))

     if step % 100 == 0:
      print(f"Step: {step}, Loss: {total_loss.item()}, Loss: {loss.item()/num_samples}, Lambda_0: {lambda_0.item()}, Lambda_2: {lambda_2.item()}")

      # Plotting the distributions
      tau_values = np.linspace(0.01, 1, 1000)
      q_values = q_tau(tau_values)
      p_values = model(torch.tensor(tau_values.reshape(-1, 1), dtype=torch.float32)).detach().numpy().flatten()*q_values
      # Evaluate p_true(τ) for the same tau values using the PyTorch function
      p_true_values = p_true_tau(tau_values)

      print(p_true_values.mean())

      # Plotting the distributions
      plt.figure(figsize=(10, 5))
      plt.plot(tau_values, p_true_values/p_true_values.mean(), label=f'p_true(τ)', color='green')
      plt.plot(tau_values, q_values/q_values.mean(), label='q(τ)', color='orange', linestyle='dashed')
      plt.plot(tau_values, p_values/p_values.mean(), label=f'p(τ) at step {step}', color='blue')
      plt.title('p(τ) vs q(τ)')
      plt.xlabel('τ')
      plt.ylabel('Probability Density')
      plt.legend()
      plt.show()
