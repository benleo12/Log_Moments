import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Assuming alpha_s is a known constant
alpha_s = 0.118
n_samp=100

from scipy.integrate import quad

# Constants
min_tau = 10**-3
max_tau = 0.1
coeff = 2 * alpha_s / (3 * np.pi)

# Define wLL and wNLL functions
def wLL(t):
    return coeff * (-4 * np.log(t) / t) * np.exp(-coeff * (2 * np.log(t)**2))

def wNLL(t):
    return coeff * ((-4 * np.log(t) + 3) / t) * np.exp(- coeff * (2 * np.log(t)**2 - 3 * np.log(t)))

# Numerical integration
CLL0, _ = quad(wLL, min_tau, max_tau)
CLL1, _ = quad(lambda t: wLL(t) * np.log(t), min_tau, max_tau)
CLL2, _ = quad(lambda t: wLL(t) * np.log(t)**2, min_tau, max_tau)

CNLL0, _ = quad(wNLL, min_tau, max_tau)
CNLL1, _ = quad(lambda t: wNLL(t) * np.log(t), min_tau, max_tau)
CNLL2, _ = quad(lambda t: wNLL(t) * np.log(t)**2, min_tau, max_tau)

print("C's",CLL0,CLL1,CLL2,CNLL0,CNLL1,CNLL2)


# Dataset of {tau_i, q_tau_i} pairs
r_i = torch.rand(n_samp)
kap_1 = -6*alpha_s/(3*np.pi)/2
kap_2 = 4*alpha_s/(3*np.pi)
tau_i = torch.exp(-np.sqrt(-torch.log(r_i)/kap_2+kap_1**2/4/kap_2**2)-kap_1/2/kap_2)
#loop until tau_i falls below LQCD/MZ (Lep) ~ 10^-4
#then sum and define it to be tau_i
#and repeat up to n_samp
#analysis.py and hemisphere.py
#generate events these will be tau

#tau_i = torch.exp(-np.sqrt(-torch.log(r_i)/kap_2))
q_tau_i = 1



# Manual integration function using Riemann sum
def mc_integration(integrand, tau_values, lambda_2, n_samp):
    integral = torch.sum(integrand)/n_samp
    return integral

# Define the integral equations using the dataset directly
def integral_equation_1_direct(lambda_0, lambda_1, lambda_2, tau_i, n_samp):
    part = (2*(kap_2+lambda_2)*torch.log(tau_i)+(kap_1+lambda_1))/(2*kap_2*torch.log(tau_i)+kap_1)
    expon = torch.exp(-1 - lambda_0 - lambda_1 * torch.log(tau_i) - lambda_2 * torch.log(tau_i)**2)
    integral = mc_integration(expon*part, tau_i, lambda_2, n_samp)
    return integral - CNLL0

def integral_equation_2_direct(lambda_0, lambda_1, lambda_2, tau_i, n_samp):
    part = (2*(kap_2+lambda_2)*torch.log(tau_i)+(kap_1+lambda_1))/(2*kap_2*torch.log(tau_i)+kap_1)
    expon = torch.exp(-1 - lambda_0 - lambda_1 * torch.log(tau_i) - lambda_2 * torch.log(tau_i)**2) * torch.log(tau_i)
    integral = mc_integration(expon*part, tau_i,lambda_2, n_samp)
    return integral - CNLL1  #(np.sqrt(3)*np.pi*np.exp(3*alpha_s/(4*np.pi))*erfc(np.sqrt(alpha_s)/2*np.sqrt(3/np.pi))/(4*np.sqrt(alpha_s)))

def integral_equation_3_direct(lambda_0, lambda_1, lambda_2, tau_i, n_samp):
    part = (2*(kap_2+lambda_2)*torch.log(tau_i)+(kap_1+lambda_1))/(2*kap_2*torch.log(tau_i)+kap_1)
    expon = torch.exp(-1 - lambda_0 - lambda_1 * torch.log(tau_i) - lambda_2 * torch.log(tau_i)**2) * torch.log(tau_i)**2
    integral = mc_integration(expon*part, tau_i, lambda_2, n_samp)
    return integral - CNLL2 #3*np.pi/(8*alpha_s)*(-2 + np.sqrt(3)*np.sqrt(alpha_s)*np.exp(3*alpha_s/(4*np.pi))*erfc(np.sqrt(alpha_s)/2*np.sqrt(3/np.pi)) )

# Initialize the Lagrange multipliers
lambda_0 = torch.tensor([-1.0], requires_grad=True)
lambda_1 = torch.tensor([-6 * alpha_s / (3 * np.pi)*0.1], requires_grad=True)
lambda_2 = torch.tensor([4 * alpha_s / (3 * np.pi)*0.0], requires_grad=True)

# Define the optimizer
optimizer = optim.Adam([lambda_1], lr=0.001)
print("kap_1 = ", kap_1)
print("kap_2 = ", kap_2)
# Optimization loop
nstep = 5000
# Lists to collect data
lambda_1_values = []
lambda_2_values = []
loss_values = []
n_samp_values = []
for step in range(1000000):
    r_i = torch.rand(n_samp)
    tau_i = torch.exp(-np.sqrt(-torch.log(r_i)/kap_2+kap_1**2/4/kap_2**2)-kap_1/2/kap_2)


    filtered_tau_i = tau_i[(tau_i >= min_tau) & (tau_i <= max_tau)]
    optimizer.zero_grad()
    loss_1 = integral_equation_1_direct(lambda_0, lambda_1, lambda_2,filtered_tau_i, n_samp)**2
    loss_2 = integral_equation_2_direct(lambda_0, lambda_1, lambda_2,filtered_tau_i, n_samp)**2
    loss_3 = integral_equation_3_direct(lambda_0, lambda_1, lambda_2,filtered_tau_i, n_samp)**2
    loss = (loss_1 + loss_2 + loss_3)  # Total loss


    if torch.isnan(loss):
     #print("sad, tau=", torch.sum(torch.isnan(tau_i)))
     #print(f"Step {step}, Loss: {loss.item()}, Lambda 0: {lambda_0.item()}, Lambda 1: {lambda_1.item()}, Lambda 2: {lambda_2.item()}")
     continue
    loss.backward()
    optimizer.step()

    if step >0 and step % (nstep+5) == 0:
        print(f"Step {step}, Loss: {loss.item()}, Lambda 0: {lambda_0.item()}, Lambda 1: {lambda_1.item()}, Lambda 2: {lambda_2.item()}")
        n_samp = int(n_samp*10)
        nstep = int(nstep/2)
        print("nstep = ", nstep)
        lambda_1_values.append(lambda_1.item())
        lambda_2_values.append(lambda_2.item())
        loss_values.append(loss.item())
        n_samp_values.append(n_samp)
    if nstep <= 5:
        break



# Plotting outside the loop

# Assuming n_samp_values and lambda_1_values are defined

plt.figure(figsize=(10, 6))
plt.plot(n_samp_values, lambda_1_values, 'o-', color='tab:red')  # Plotting with markers and lines
plt.xscale('log')  # Applying log scale to the x-axis
plt.xlabel(r'$n_{samp}$', fontsize=14)
plt.ylabel(r'$\lambda_1$', color='tab:red', fontsize=14)
plt.title('Log moment progression', fontsize=16)
plt.grid(True, which="both", ls="--")  # Adding grid lines for readability
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(n_samp_values, lambda_2_values, 'o-', color='tab:red')  # Plotting with markers and lines
plt.xscale('log')  # Applying log scale to the x-axis
plt.xlabel(r'$n_{samp}$', fontsize=14)
plt.ylabel(r'$\lambda_2$', color='tab:red', fontsize=14)
plt.title('Log moment progression', fontsize=16)
plt.grid(True, which="both", ls="--")  # Adding grid lines for readability
plt.show()

# Log-log plot for loss
plt.figure(figsize=(10, 6))
plt.loglog(n_samp_values, loss_values, color='tab:blue')
plt.xlabel(r'$n_{samp}$', fontsize=14)
plt.ylabel('Loss', color='tab:blue', fontsize=14)
plt.title('Loss progression', fontsize=16)
plt.tick_params(axis='y', labelcolor='tab:blue')
plt.grid(False, which="both", ls="--")  # Adding grid lines for readability
plt.tight_layout()
plt.show()


# Print final values of the Lagrange multipliers

print(f"Final Lambda 0: {lambda_0.item()}, Final Lambda 1: {lambda_1.item()}, Final Lambda 2: {lambda_2.item()}")
