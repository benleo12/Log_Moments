import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Constants
alpha_s = 0.1
pi = np.pi

# Define the prior distribution q(τ)
def q_tau(tau):
    return (2 * alpha_s / (3 * pi * tau)) * (-4 * np.log(tau))

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
        self.activation = nn.ReLU()

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

# Optimizer and Loss Function
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training the model
num_steps = 1000
num_samples = 1000

for step in range(num_steps):
    tau_samples = torch.rand(num_samples, 1) * (1.0 - 0.5) + 0.5 # samples between 0.5 and 1.0
    q_tau_values = torch.tensor(q_tau(tau_samples.numpy()), dtype=torch.float32)

    optimizer.zero_grad()
    p_tau_values = model(tau_samples)
    loss = torch.mean(p_tau_values * (torch.log(p_tau_values) - torch.log(q_tau_values)))

    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step: {step}, Loss: {loss.item()}, Loss: {loss.item()}")

# Plotting the distributions
tau_values = np.linspace(0.5, 1.0, 1000)
p_values = model(torch.tensor(tau_values.reshape(-1, 1), dtype=torch.float32)).detach().numpy().flatten()
q_values = q_tau(tau_values)

plt.figure(figsize=(10, 5))
plt.plot(tau_values, p_values, label='p(τ)', color='blue')
plt.plot(tau_values, q_values, label='q(τ)', color='orange', linestyle='dashed')
plt.title('p(τ) vs q(τ)')
plt.xlabel('τ')
plt.ylabel('Probability Density')
plt.legend()
plt.show()


