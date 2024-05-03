import torch
import torch.optim as optim
import numpy as np

# Assuming alpha_s is a known constant
alpha_s = 0.3
steps=10000

# Dataset of {tau_i, q_tau_i} pairs
r_i = torch.rand(steps)
kap_1 = 0.05
kap_2 = 4*alpha_s/(3*torch.pi)
tau_i = torch.exp(-torch.sqrt(-torch.log(r_i)/kap_2+kap_1**2/4/kap_2**2)-kap_1/2/kap_2)
#tau_i = torch.exp(-torch.sqrt(-torch.log(r_i)/kap_2))
q_tau_i = 1



# Manual integration function using Riemann sum
def mc_integration(integrand, tau_values, lambda_2):
#    C= 4*2*alpha_s/(3*np.pi)
#    integral = torch.mean(integrand)*C/2*np.log(1e-5)**2
    integral = torch.mean(integrand)
    return integral

# Define the integral equations using the dataset directly
def integral_equation_1_direct(lambda_0, lambda_1, lambda_2):
    part = (kap_2+lambda_2)/kap_2+(lambda_1+kap_1)/(2*kap_2) + (kap_2*lambda_1-kap_1*lambda_2)/(kap_2*(kap_1+2*kap_2*torch.log(tau_i)))
    expon = torch.exp(-1 - lambda_0 - lambda_1 * torch.log(tau_i) - lambda_2 * torch.log(tau_i)**2)
    integral = mc_integration(expon*part, tau_i, lambda_2)
    return integral - 1

def integral_equation_2_direct(lambda_0, lambda_1, lambda_2):
    part = (kap_2+lambda_2)/kap_2+(lambda_1+kap_1)/(2*kap_2) + (kap_2*lambda_1-kap_1*lambda_2)/(kap_2*(kap_1+2*kap_2*torch.log(tau_i)))
    expon = torch.exp(-1 - lambda_0 - lambda_1 * torch.log(tau_i) - lambda_2 * torch.log(tau_i)**2) * torch.log(tau_i)
    integral = mc_integration(expon*part, tau_i,lambda_2)
    return integral + np.sqrt(3) * torch.pi / (4 * np.sqrt(alpha_s))

def integral_equation_3_direct(lambda_0, lambda_1, lambda_2):
    part = (kap_2+lambda_2)/kap_2+(lambda_1+kap_1)/(2*kap_2) + (kap_2*lambda_1-kap_1*lambda_2)/(kap_2*(kap_1+2*kap_2*torch.log(tau_i)))
    expon = torch.exp(-1 - lambda_0 - lambda_1 * torch.log(tau_i) - lambda_2 * torch.log(tau_i)**2) * torch.log(tau_i)**2
    integral = mc_integration(expon*part, tau_i, lambda_2)
    return integral - (3 * torch.pi) / (4 * alpha_s)

# Initialize the Lagrange multipliers
lambda_0 = torch.tensor([-1.0], requires_grad=True)
lambda_1 = torch.tensor([(-8*np.sqrt(alpha_s)/(torch.pi * np.sqrt(3)))*0.1], requires_grad=True)
lambda_2 = torch.tensor([4 * alpha_s / (3 * torch.pi)*0.1], requires_grad=True)

# Define the optimizer
optimizer = optim.Adam([lambda_1, lambda_2], lr=0.005)
print("kap_1 = ", kap_1)
print("kap_2 = ", kap_2)
# Optimization loop
for step in range(1000000):
    r_i = torch.rand(steps)
    tau_i = torch.exp(-torch.sqrt(-torch.log(r_i)/kap_2+kap_1**2/4/kap_2**2)-kap_1/2/kap_2)


    optimizer.zero_grad()
    loss_1 = integral_equation_1_direct(lambda_0, lambda_1, lambda_2)**2
    loss_2 = integral_equation_2_direct(lambda_0, lambda_1, lambda_2)**2
    loss_3 = integral_equation_3_direct(lambda_0, lambda_1, lambda_2)**2
    loss = loss_1 + loss_2 + loss_3  # Total loss
    if torch.isnan(loss):
     #print("sad, tau=", torch.sum(torch.isnan(tau_i)))
     #print(f"Step {step}, Loss: {loss.item()}, Lambda 0: {lambda_0.item()}, Lambda 1: {lambda_1.item()}, Lambda 2: {lambda_2.item()}")
     continue
    loss.backward()
    optimizer.step()

    if step >0 and step % 1000 == 0:
        print(f"Step {step}, Loss: {loss.item()}, Lambda 0: {lambda_0.item()}, Lambda 1: {lambda_1.item()}, Lambda 2: {lambda_2.item()}")
        steps *= 2

# Print final values of the Lagrange multipliers
print(f"Final Lambda 0: {lambda_0.item()}, Final Lambda 1: {lambda_1.item()}, Final Lambda 2: {lambda_2.item()}")
