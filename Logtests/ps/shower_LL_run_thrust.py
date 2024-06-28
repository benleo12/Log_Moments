import argparse
print("Script started")
# other imports and code follow


parser = argparse.ArgumentParser(description='Process events and analyze thrust.')
parser.add_argument('--e', type=int, default=20000, help='Number of samples to process')
parser.add_argument('--n_step', type=int, default=100, help='Number of steps in the process')
parser.add_argument('--step_frac', type=float, default=1.1, help='Fractional step increase per iteration')
parser.add_argument('--samp_fac', type=float, default=10, help='Sample factor for scaling')
parser.add_argument('--asif', type=float, default=0.02, help='alphas limit')
parser.add_argument('--min_t', type=float, default=1e-8, help='alphas limit')
# Debugging: Verify this line is executed
print("Debug: Adding arguments complete")
import sys
print(sys.argv)

# Parse arguments
args = parser.parse_args()
n_samp = args.e
n_step = args.n_step
step_frac = args.step_frac
samp_fac = args.samp_fac
asif = args.asif
min_tau = args.min_t


# Define a function to save parameters to a file
def save_params(lambda_2, asif, filename="params.txt"):
    with open(filename, 'w') as f:
        f.write(f"lambda_2: {lambda_2}\n")
        f.write(f"asif: {asif}\n")

print("Script started")


import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import subprocess
import csv
import time
import os
import nll_torch
from qcd import AlphaS, NC, TR, CA, CF
from scipy.integrate import quad



# Assuming alpha_s is a known constant
alpha_s = 0.118

# Integration range
#min_tau = 10**-10
max_tau = 0.001
coeff = 2 * alpha_s / (3 * np.pi)

alphas = [ AlphaS(91.1876,asif,0), AlphaS(91.1876,asif,0) ]
analytics = nll_torch.NLL(alphas,a=1,b=1,t=91.1876**2)


# Define wLL and wNLL functions
def wLL(tau):
     partC = (analytics.Rp(tau))/tau
     exponC = np.exp(-analytics.R_L(tau))
     return partC*exponC

def torch_quad(func, a, b, func_mul=None, func_mul2=None, num_steps=10000000):
    # Create logarithmically spaced points
    x = torch.logspace(torch.log10(a), torch.log10(b), steps=num_steps, dtype=torch.float64)

    # Calculate differential elements, adjusted for log spacing
    dx = (x[1:] - x[:-1])  # More accurate differential using actual spacing between points

    # Evaluate the function at these points
    y = func(x[:-1])  # Evaluate function at left endpoints (or midpoints if preferred)

    # Apply any additional multiplicative functions if provided
    if func_mul is not None:
        y *= func_mul(x[:-1])
    if func_mul2 is not None:
        y *= func_mul2(x[:-1])

    # Debug print to check values of y*dx
    y_dx = y * dx
    print("YDX:", torch.sum(y_dx))

    # Sum up y*dx to get the integral approximation
    integral = torch.sum(y_dx)
    return integral


def torch_quad_old(func, a, b, func_mul=None, func_mul2=None, num_steps=100000):
    x = torch.logspace(torch.log10(a), torch.log10(b), steps=num_steps, dtype=torch.float64)
    dx = np.log(10)*(torch.log10(b) - torch.log10(a))*x / num_steps
    y = func(x)
    #print("y before applying func_mul:", y)
    if func_mul is not None:
        y = y*func_mul(x)
    if func_mul2 is not None:
        y = y*func_mul2(x)
    #print("y after applying func_mul:", y)
    print("YDX:",torch.sum(y * dx))
    return torch.sum(y * dx)



min_tau = torch.tensor(min_tau)
max_tau = torch.tensor(max_tau)

CLL0 = torch_quad(wLL, min_tau, max_tau)
CLL = torch_quad(wLL, min_tau, max_tau, func_mul=analytics.R_L)
#CLL = torch_quad(wLL, min_tau, max_tau, func_mul=torch.log)#/CLL0

print("CLL0=",CLL0)
print("CLL=",CLL)


def run_pypy_script(pypy_script_path):
    """Runs the PyPy script with specified flags that generates the CSV file."""
    # Define the additional flags as a list
    flags = ['-e',str(n_samp), '-A',str(asif), '-n','1', '-O','0','-b','1']

    try:
        # Include the additional flags in the subprocess call
        subprocess.check_call(['pypy', pypy_script_path] + flags)
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


# Manual integration function using Riemann sum
def mc_integration(integrand, tau_values, n_samp):
    integral = torch.sum(integrand)/n_samp
    return integral

# Define the integral equations using the dataset directly
def integral_equation_1_direct(lambda_0, lambda_1, lambda_2, tau_i, n_samp):
    Rpt = analytics.Rp(tau_i)
    logFt = analytics.logF(tau_i)
    FpFt = analytics.FpF(tau_i)
    RNNLLpt = analytics.RNNLLp(tau_i)
    part = (CLL*Rpt)/((CLL-lambda_2)*Rpt)
    expon = torch.exp( -  lambda_2*analytics.R_L(tau_i))
    moment = 1
    integral = mc_integration(expon*part*moment, tau_i, n_samp)/mc_integration(expon*part, tau_i, n_samp)
    integral_0 = mc_integration(torch.ones(len(tau_i)), tau_i, n_samp)
    print("check unitary:",integral)
    print("check unitary 1:",integral_0)
    return integral - CLL0

def integral_equation_2_direct(lambda_0,lambda_1, lambda_2, tau_i, n_samp):
    Rpt = analytics.Rp(tau_i)
    logFt = analytics.logF(tau_i)
    FpFt = analytics.FpF(tau_i)
    RNNLLpt = analytics.RNNLLp(tau_i)
    part = (CLL)/((CLL-lambda_2))
    expon = torch.exp( -  lambda_2*analytics.R_L(tau_i))
    moment = analytics.R_L(tau_i)
    #moment = torch.log(tau_i)**2
    integral = mc_integration(expon*part*moment, tau_i, n_samp)/mc_integration(expon*part, tau_i, n_samp)
    return integral - CLL


# Initialize the Lagrange multipliers
lambda_0 = torch.tensor([0.2819207*0], requires_grad=True)
lambda_1 = torch.tensor([0.3137269*0], requires_grad=True)
lambda_2 = torch.tensor([0.5], requires_grad=True)

# Define the optimizer
optimizer = optim.Adam([lambda_2], lr=0.01)

# Optimization loop
# Lists to collect data
lambda_1_values = []
lambda_2_values = []
loss_values = []
n_samp_values = []
min_loss = 1e-5
no_decrease_counter = 0
max_no_decrease_steps = 10

# Generate data once
pypy_script_path = os.path.expanduser('~/Dropbox/LogMoments/Logtests/ps/dire.py')
thrust_path = os.path.expanduser('~/Dropbox/LogMoments/Logtests/ps/thrust_values.csv')

if run_pypy_script(pypy_script_path):
    if os.path.exists(thrust_path):
        tau_i = read_csv_to_torch(thrust_path)
    else:
        print(f"CSV file not found: {thrust_path}")
        raise RuntimeError("Data generation failed. Exiting.")
else:
    print("PyPy script execution failed.")
    raise RuntimeError("Data generation failed. Exiting.")

filtered_tau_0 = tau_i[(tau_i <= min_tau)]
filtered_tau_i = tau_i[(tau_i >= min_tau) & (tau_i <= max_tau)]
print("zero taus = ", len(filtered_tau_0))

for step in range(1000000):

    print(f"Running optimization step {step+1}")
    
    filtered_tau_i = tau_i[(tau_i >= min_tau) & (tau_i <= max_tau)]
    optimizer.zero_grad()
    loss_1 = torch.abs(integral_equation_1_direct(lambda_0, lambda_1, lambda_2, filtered_tau_i, n_samp))
    loss_2 = torch.abs(integral_equation_2_direct(lambda_0, lambda_1, lambda_2, filtered_tau_i, n_samp))
    loss = loss_2


    if torch.isnan(loss):
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

    # Save parameters if loss stops decreasing
    if step > 0 and loss.item() < min_loss:
        save_params(lambda_2.item(), asif)
        break



#
# # Plotting outside the loop
#
# # Assuming n_samp_values and lambda_1_values are defined
#
# plt.figure(figsize=(10, 6))
# plt.plot(n_samp_values, lambda_1_values, 'o-', color='tab:red')  # Plotting with markers and lines
# plt.xscale('log')  # Applying log scale to the x-axis
# plt.xlabel(r'$n_{samp}$', fontsize=14)
# plt.ylabel(r'$\lambda_1$', color='tab:red', fontsize=14)
# plt.title('Log moment progression', fontsize=16)
# plt.grid(True, which="both", ls="--")  # Adding grid lines for readability
# plt.show()
#
# plt.figure(figsize=(10, 6))
# plt.plot(n_samp_values, lambda_2_values, 'o-', color='tab:red')  # Plotting with markers and lines
# plt.xscale('log')  # Applying log scale to the x-axis
# plt.xlabel(r'$n_{samp}$', fontsize=14)
# plt.ylabel(r'$\lambda_2$', color='tab:red', fontsize=14)
# plt.title('Log moment progression', fontsize=16)
# plt.grid(True, which="both", ls="--")  # Adding grid lines for readability
# plt.show()
#
# # Log-log plot for loss
# plt.figure(figsize=(10, 6))
# plt.loglog(n_samp_values, loss_values, color='tab:blue')
# plt.xlabel(r'$n_{samp}$', fontsize=14)
# plt.ylabel('Loss', color='tab:blue', fontsize=14)
# plt.title('Loss progression', fontsize=16)
# plt.tick_params(axis='y', labelcolor='tab:blue')
# plt.grid(False, which="both", ls="--")  # Adding grid lines for readability
# plt.tight_layout()
# plt.show()


# Print final values of the Lagrange multipliers

print(f"Final Lambda 0: {lambda_0.item()}, Final Lambda 1: {lambda_1.item()}, Final Lambda 2: {lambda_2.item()}")
