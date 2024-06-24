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
import nll_torch
from qcd import AlphaS, NC, TR, CA, CF
from scipy.integrate import quad

parser = argparse.ArgumentParser(description='Process events and analyze thrust.')
parser.add_argument('--n_samp', type=int, default=40000, help='Number of samples to process')
parser.add_argument('--n_step', type=int, default=1000, help='Number of samples to process')
parser.add_argument('--step_frac', type=float, default=1.5, help='Number of samples to process')
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
max_tau = 0.99
coeff = 2 * alpha_s / (3 * np.pi)

alphas = [ AlphaS(91.2,0.118,0), AlphaS(91.2,0.118,1) ]
analytics = nll_torch.NLL(alphas,a=1,b=1,t=91.2**2)


# Define wLL and wNLL functions
def wLL(tau):
     partC = (analytics.Rp(tau))/tau
     exponC = np.exp(-analytics.R_L(tau))
#    partC = ((analytics.RNNLLp(tau)-analytics.FpF(tau)))/tau
#    exponC = np.exp( - (analytics.R_SL(tau)-analytics.logF(tau)) )
     return partC*exponC 
    


def wNLL(tau):
     partC = (analytics.Rp(tau) + (analytics.RNNLLp(tau)-analytics.FpF(tau)))/tau
     exponC = np.exp( - (analytics.R_SL(tau)-analytics.logF(tau)) -  analytics.R_L(tau))
     return partC*exponC


def torch_quad(func, a, b, num_steps=1000000, func_mul=None, func_mul2=None, func_mul3=None):
    x = torch.linspace(a, b, steps=num_steps, dtype=torch.float32)
    dx = (b - a) / num_steps
    y = func(x)
    if func_mul is not None:
        y *= func_mul(x)
    if func_mul2 is not None:
        y *= func_mul2(x)
    if func_mul3 is not None:
        y *= func_mul3(x)
    return torch.sum(y * dx)



min_tau = torch.tensor(min_tau)
max_tau = torch.tensor(max_tau)

CLL0 = torch_quad(wLL, min_tau, max_tau)
CLL1 = torch_quad(wLL, min_tau, max_tau, func_mul=torch.log)
CLL2 = torch_quad(wLL, min_tau, max_tau, func_mul=torch.log, func_mul2=torch.log)

CNLL0 = torch_quad(wNLL, min_tau, max_tau)
CNLL1 = torch_quad(wNLL, min_tau, max_tau, func_mul=torch.log)
CNLL2 = torch_quad(wNLL, min_tau, max_tau, func_mul=torch.log, func_mul2=torch.log)
CNLL3 = torch_quad(wNLL, min_tau, max_tau, func_mul=torch.log, func_mul2=torch.log, func_mul3=torch.log)

print("CNLL0=",CNLL0)


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



# Define the range
taus = np.linspace(-3, -0.01, 100)


kap_1 = -6*alpha_s/(3*np.pi)
kap_2 = 4*alpha_s/(3*np.pi)

# Manual integration function using Riemann sum
def mc_integration(integrand, tau_values, n_samp):
    integral = torch.sum(integrand)/n_samp
    return integral

# Define the integral equations using the dataset directly
def integral_equation_1_direct(lambda_0, lambda_1, lambda_2,lambda_3, tau_i, n_samp):
    Rpt = analytics.Rp(tau_i)
    logFt = analytics.logF(tau_i)
    FpFt = analytics.FpF(tau_i)
    RNNLLpt = analytics.RNNLLp(tau_i) 
    part = (CNLL3*3*torch.log(tau_i)**2+ CNLL2*Rpt + CNLL1*(RNNLLpt-FpFt))/((CNLL2-lambda_2)*Rpt+(CNLL1-lambda_1)*(RNNLLpt-FpFt)+(CNLL3-lambda_3)*3*torch.log(tau_i)**2)
    expon = torch.exp(lambda_0 - lambda_1*(analytics.R_SL(tau_i)-analytics.logF(tau_i)) -  lambda_2*analytics.R_L(tau_i) - lambda_3*torch.log(tau_i)**3)
    integral = mc_integration(expon*part, tau_i, n_samp)
    integral_0 = mc_integration(torch.ones(len(tau_i)), tau_i, n_samp)
    print("check unitary:",integral)
    print("check unitary 1:",integral_0)
    return integral - CNLL0

def integral_equation_2_direct(lambda_0, lambda_1, lambda_2, lambda_3, tau_i, n_samp):
    Rpt = analytics.Rp(tau_i)
    logFt = analytics.logF(tau_i)
    FpFt = analytics.FpF(tau_i)
    RNNLLpt = analytics.RNNLLp(tau_i) 
    part = (CNLL3*3*torch.log(tau_i)**2+ CNLL2*Rpt + CNLL1*(RNNLLpt-FpFt))/((CNLL2-lambda_2)*Rpt+(CNLL1-lambda_1)*(RNNLLpt-FpFt)+(CNLL3-lambda_3)*3*torch.log(tau_i)**2)
    expon = torch.exp(lambda_0 - lambda_1*(analytics.R_SL(tau_i)-analytics.logF(tau_i)) -  lambda_2*analytics.R_L(tau_i) - lambda_3*torch.log(tau_i)**3)*torch.log(tau_i)
    integral = mc_integration(expon*part, tau_i, n_samp)
    return integral - CNLL1

def integral_equation_3_direct(lambda_0, lambda_1, lambda_2, lambda_3,tau_i, n_samp):
    Rpt = analytics.Rp(tau_i)
    logFt = analytics.logF(tau_i)
    FpFt = analytics.FpF(tau_i)
    RNNLLpt = analytics.RNNLLp(tau_i)
    part = (CNLL3*3*torch.log(tau_i)**2+ CNLL2*Rpt + CNLL1*(RNNLLpt-FpFt))/((CNLL2-lambda_2)*Rpt+(CNLL1-lambda_1)*(RNNLLpt-FpFt)+(CNLL3-lambda_3)*3*torch.log(tau_i)**2)
    expon = torch.exp(lambda_0 - lambda_1*(analytics.R_SL(tau_i)-analytics.logF(tau_i)) -  lambda_2*analytics.R_L(tau_i) - lambda_3*torch.log(tau_i)**3)*torch.log(tau_i)**2
    integral = mc_integration(expon*part, tau_i, n_samp)
    return integral - CNLL2

def integral_equation_4_direct(lambda_0, lambda_1, lambda_2,lambda_3, tau_i, n_samp):
    Rpt = analytics.Rp(tau_i)
    logFt = analytics.logF(tau_i)
    FpFt = analytics.FpF(tau_i)
    RNNLLpt = analytics.RNNLLp(tau_i)
    part = (CNLL3*3*torch.log(tau_i)**2+ CNLL2*Rpt + CNLL1*(RNNLLpt-FpFt))/((CNLL2-lambda_2)*Rpt+(CNLL1-lambda_1)*(RNNLLpt-FpFt)+(CNLL3-lambda_3)*3*torch.log(tau_i)**2)
    expon = torch.exp(lambda_0 - lambda_1*(analytics.R_SL(tau_i)-analytics.logF(tau_i)) -  lambda_2*analytics.R_L(tau_i) - lambda_3*torch.log(tau_i)**3)*torch.log(tau_i)**3
    integral = mc_integration(expon*part, tau_i, n_samp)
    return integral - CNLL3

# Initialize the Lagrange multipliers
lambda_0 = torch.tensor([0.0], requires_grad=True)
lambda_1 = torch.tensor([0.0], requires_grad=True)
lambda_2 = torch.tensor([0.0], requires_grad=True)
lambda_3 = torch.tensor([0.0], requires_grad=True) 
# Define the optimizer
optimizer = optim.Adam([lambda_0,lambda_1,lambda_2,lambda_3], lr=0.01)

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
    pypy_script_path = os.path.expanduser('~/Dropbox/LogMoments/tutorials/ps/shower_torch_run_phys.py')
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
    filtered_tau_0 = tau_i[(tau_i <= 10**-9)]
    print("zero taus = ",len(filtered_tau_0))
    filtered_tau_i = tau_i[(tau_i >= min_tau) & (tau_i <= max_tau)]
    optimizer.zero_grad()
    loss_1 = torch.abs(integral_equation_1_direct(lambda_0, lambda_1, lambda_2,lambda_3, filtered_tau_i, n_samp))
    loss_2 = torch.abs(integral_equation_2_direct(lambda_0, lambda_1, lambda_2,lambda_3, filtered_tau_i, n_samp))
    loss_3 = torch.abs(integral_equation_3_direct(lambda_0, lambda_1, lambda_2, lambda_3, filtered_tau_i, n_samp))
    loss_4 = torch.abs(integral_equation_4_direct(lambda_0, lambda_1, lambda_2, lambda_3, filtered_tau_i, n_samp))
    loss = loss_1 + loss_2 + loss_3+loss_4  # Total loss


    if torch.isnan(loss):
     continue
    loss.backward()
    optimizer.step()

    print(f"Step {step}, Loss: {loss.item()}, Lambda 0: {lambda_0.item()}, Lambda 1: {lambda_1.item()}, Lambda 2: {lambda_2.item()}, Lambda 3: {lambda_3.item()}")

    if step >0 and step % (n_step+5) == 0:
        print(f"Step {step}, Loss: {loss.item()}, Lambda 0: {lambda_0.item()}, Lambda 1: {lambda_1.item()}, Lambda 2: {lambda_2.item()}, Lambda 3: {lambda_3.item()}")
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
