import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import argparse
import subprocess
import csv
import time
import os
from scipy.integrate import quad

parser = argparse.ArgumentParser(description='Process events and analyze thrust.')
parser.add_argument('--n_samp', type=int, default=100000, help='Number of samples to process')
parser.add_argument('--n_step', type=int, default=1000, help='Number of samples to process')
parser.add_argument('--step_frac', type=float, default=1.1, help='Number of samples to process')
parser.add_argument('--samp_fac', type=float, default=10, help='Number of samples to process')
# Parse arguments
args = parser.parse_args()

n_samp = args.n_samp
n_step = args.n_step
step_frac = args.step_frac
samp_fac = args.samp_fac

# Assuming alpha_s is a known constant
alpha_s = 0.118

# Integration range
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



def run_pypy_script(pypy_script_path):
    """Runs the PyPy script that generates the CSV file."""
    try:
        subprocess.check_call(['pypy', pypy_script_path, '--n_samp', str(n_samp)])
    except subprocess.CalledProcessError as e:
        print("An error occurred while running the PyPy script:", e)
        return False
    return True


def read_csv_to_torch(csv_file_path):
    """Reads the CSV file and converts it to a PyTorch tensor."""
    with open(csv_file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        tau_values = [float(row[0]) for row in csv_reader]
    return torch.tensor(tau_values)


kap_1 = -6*alpha_s/(3*np.pi)
kap_2 = 4*alpha_s/(3*np.pi)

# Manual integration function using Riemann sum
def mc_integration(integrand, tau_values, n_samp):
    integral = torch.sum(integrand)/n_samp
    return integral

# Define the integral equations using the dataset directly
def integral_equation_1_direct(lambda_0, lambda_1, lambda_2, tau_i, n_samp):
    #part = (2*(kap_2+lambda_2)*torch.log(tau_i)+(kap_1+lambda_1))/(2*kap_2*torch.log(tau_i)+kap_1)
    part = (2*CNLL2*torch.log(tau_i)+CNLL1)/(2*(CNLL2-lambda_2)*torch.log(tau_i)+(CNLL1-lambda_1))
    expon = torch.exp(-lambda_0 - lambda_1 * torch.log(tau_i) - lambda_2 * torch.log(tau_i)**2)
    integral = mc_integration(expon*part, tau_i, n_samp)
    print("integral=",integral)
    return integral - CNLL0

def integral_equation_2_direct(lambda_0, lambda_1, lambda_2, tau_i, n_samp):
    part = (2*CNLL2*torch.log(tau_i)+CNLL1)/(2*(CNLL2-lambda_2)*torch.log(tau_i)+(CNLL1-lambda_1))
    expon = torch.exp(-lambda_0 - lambda_1 * torch.log(tau_i) - lambda_2 * torch.log(tau_i)**2) * torch.log(tau_i)
    integral = mc_integration(expon*part, tau_i, n_samp)
    return integral - CNLL1

def integral_equation_3_direct(lambda_0, lambda_1, lambda_2, tau_i, n_samp):
    part = (2*CNLL2*torch.log(tau_i)+CNLL1)/(2*(CNLL2-lambda_2)*torch.log(tau_i)+(CNLL1-lambda_1))
    expon = torch.exp(-lambda_0 - lambda_1 * torch.log(tau_i) - lambda_2 * torch.log(tau_i)**2) * torch.log(tau_i)**2
    integral = mc_integration(expon*part, tau_i, n_samp)
    return integral - CNLL2

# Initialize the Lagrange multipliers
lambda_0 = torch.tensor([-0.0], requires_grad=True)
lambda_1 = torch.tensor([-6 * alpha_s / (3 * np.pi)*0.0], requires_grad=True)
lambda_2 = torch.tensor([4 * alpha_s / (3 * np.pi)*0.0], requires_grad=True)

# Define the optimizer
optimizer = optim.Adam([lambda_1,lambda_2], lr=0.001)
print("kap_1 = ", kap_1)
print("kap_2 = ", kap_2)
# Optimization loop
# Lists to collect data
lambda_1_values = []
lambda_2_values = []
loss_values = []
n_samp_values = []
for step in range(1000000):

    print(f"Running optimization step {step+1}")
    pypy_script_path = os.path.expanduser('~/Dropbox/LogMoments/tutorials/ps/shower_torch.py')
    thrust_path = os.path.expanduser('~/Dropbox/LogMoments/tutorials/ps/thrust_values.csv')

    if run_pypy_script(pypy_script_path):
        if os.path.exists(thrust_path):
            tau_i = read_csv_to_torch(thrust_path)
        #    print("TAUS = ", tau_i)
        else:
            print(f"CSV file not found: {thrust_path}")
    else:
        print("PyPy script execution failed.")
        break  # Stop the loop if PyPy script fails to run

    filtered_tau_i = tau_i[(tau_i >= min_tau) & (tau_i <= max_tau)]
    optimizer.zero_grad()
    loss_1 = torch.abs(integral_equation_1_direct(lambda_0, lambda_1, lambda_2, filtered_tau_i, n_samp))
    loss_2 = torch.abs(integral_equation_2_direct(lambda_0, lambda_1, lambda_2, filtered_tau_i, n_samp))
    loss_3 = torch.abs(integral_equation_3_direct(lambda_0, lambda_1, lambda_2, filtered_tau_i, n_samp))
    loss = loss_1 + loss_2 + loss_3  # Total loss


    if torch.isnan(loss):
     #print("sad, tau=", torch.sum(torch.isnan(tau_i)))
     #print(f"Step {step}, Loss: {loss.item()}, Lambda 0: {lambda_0.item()}, Lambda 1: {lambda_1.item()}, Lambda 2: {lambda_2.item()}")
     continue
    loss.backward()
    optimizer.step()

    print(f"Step {step}, Loss: {loss.item()}, Lambda 0: {lambda_0.item()}, Lambda 1: {lambda_1.item()}, Lambda 2: {lambda_2.item()}")

    if step >0 and step % (n_step+5) == 0:
        print(f"Step {step}, Loss: {loss.item()}, Lambda 0: {lambda_0.item()}, Lambda 1: {lambda_1.item()}, Lambda 2: {lambda_2.item()}")
        n_samp = int(n_samp*samp_fac)
        n_step = int(n_step/step_frac)
        print("n_step = ", n_step)
        lambda_1_values.append(lambda_1.item())
        lambda_2_values.append(lambda_2.item())
        loss_values.append(loss.item())
        n_samp_values.append(n_samp)
    if n_step <= 5:
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
