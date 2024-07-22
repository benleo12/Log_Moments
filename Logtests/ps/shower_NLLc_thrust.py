import argparse
print("Script started")
# other imports and code follow


parser = argparse.ArgumentParser(description='Process events and analyze thrust.')
parser.add_argument('--e', type=int, default=20000, help='Number of samples to process')
parser.add_argument('--n_step', type=int, default=100, help='Number of steps in the process')
parser.add_argument('--step_frac', type=float, default=1, help='Fractional step increase per iteration')
parser.add_argument('--samp_fac', type=float, default=1, help='Sample factor for scaling')
parser.add_argument('--asif', type=float, default=0.02, help='alphas limit')
parser.add_argument('--min_t', type=float, default=1e-15, help='alphas limit')
parser.add_argument('--max_t', type=float, default=0.9999, help='alphas limit')
parser.add_argument('--lam_LL', type=float, default=0.0, help='alphas limit')
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
max_tau = args.max_t
lam_LL = args.lam_LL


# Define a function to save parameters to a file
def save_params(lambda_2, asif, n_samp):
    """Saves parameters to a file with the number of events in the filename."""
    filename = f"params_LL_{n_samp}.txt"
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

torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

# Set the number of threads used for intra-op parallelism
torch.set_num_threads(10)

# Set the number of threads used for inter-op parallelism
torch.set_num_interop_threads(10)
# Assuming alpha_s is a known constant
alpha_s = 0.118

# Integration range
#min_tau = 10**-10
#max_tau = 0.003

alphas = [ AlphaS(91.1876,asif,0), AlphaS(91.1876,asif,1) ]
analytics = nll_torch.NLL(alphas,a=1,b=1,t=91.1876**2)


# Define wLL and wNLL functions
def wLL(tau):
     partC = (analytics.Rp(tau)+analytics.RNLLcp(tau))/tau
     exponC = np.exp(-analytics.R_L(tau)-analytics.R_NLLc(tau))
     return partC*exponC

def torch_quad(func, a, b, func_mul=None, func_mul2=None, num_steps=1000000):
    x = torch.logspace(torch.log10(a), torch.log10(b), steps=num_steps, dtype=torch.float64)
    dx = (x[1:]-x[:-1])
    y = (func(x[1:])+func(x[:-1]))/2.
    if func_mul is not None:
        y *= (func_mul(x[1:])+func_mul(x[:-1]))/2.
    if func_mul2 is not None:
        y *= (func_mul2(x[1:])+func_mul2(x[:-1]))/2.
    y_dx = y * dx
    integral = torch.sum(y_dx)
    return integral


def run_pypy_script(pypy_script_path, n_samp, asif):
    """Runs the PyPy script with specified flags that generates the CSV file."""
    # Define the additional flags as a list
    flags = ['-e', str(n_samp), '-A', str(asif), '-n', '1', '-O', '1', '-b', '1']
    
    # Create the filename based on flags and ensure it includes "LL"
    filename = f"thrust_e{n_samp}_A{asif}_NLLc.csv"
    
    # Check if the file already exists
    if os.path.exists(filename):
        print(f"File {filename} already exists. Not overwriting.")
        return filename

    try:
        # Include the additional flags in the subprocess call
        subprocess.check_call(['pypy', pypy_script_path] + flags)
        
        # Check if the expected output file exists and rename it
        if os.path.exists("thrust_NLLc.csv"):
            os.rename("thrust_NLLc.csv", filename)
        else:
            print("Expected output file 'thrust_LL.csv' not found.")
            return None
        
    except subprocess.CalledProcessError as e:
        print("An error occurred while running the PyPy script:", e)
        return None
    
    return filename

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
def integral_equation_1_direct(lambda_0,lambda_1, lambda_2, tau_i, n_samp):
    Rpt = analytics.Rp(tau_i)
    logFt = analytics.logF(tau_i)
    FpFt = analytics.FpF(tau_i)
    RNLLpt = analytics.RNLLcp(tau_i)
    part = (CLL*Rpt + CNLL*(RNLLpt))/((CLL-lambda_1)*Rpt+(CNLL-lambda_2)*(RNLLpt))
    expon = torch.exp(-lambda_1*analytics.R_L(tau_i)-lambda_2*analytics.R_NLLc(tau_i))
    moment = analytics.R_L(tau_i)
    integral = mc_integration(expon*part*moment, tau_i, n_samp)/mc_integration(expon*part, tau_i, n_samp)
    return integral - CLL


def integral_equation_2_direct(lambda_0,lambda_1, lambda_2, tau_i, n_samp):
    Rpt = analytics.Rp(tau_i)
    logFt = analytics.logF(tau_i)
    FpFt = analytics.FpF(tau_i)
    RNLLpt = analytics.RNLLcp(tau_i)
    part = (CLL*Rpt + CNLL*(RNLLpt))/((CLL-lambda_1)*Rpt+(CNLL-lambda_2)*(RNLLpt))
    expon = torch.exp(-lambda_1*analytics.R_L(tau_i)-lambda_2*analytics.R_NLLc(tau_i))
    moment = analytics.R_NLLc(tau_i)
    integral = mc_integration(expon*part*moment, tau_i, n_samp)/mc_integration(expon*part, tau_i, n_samp)
    return integral - CNLL


# Initialize the Lagrange multipliers
lambda_0 = torch.tensor([0.0], requires_grad=True)
lambda_2 = torch.tensor([0.0], requires_grad=True)
lambda_1 = torch.tensor([lam_LL], requires_grad=True)

# Define the optimizer
optimizer = optim.Adam([lambda_2], lr=0.01)

# Optimization loop
# Lists to collect data
lambda_1_values = []
lambda_2_values = []
loss_values = []
n_samp_values = []
min_loss = 1e-07
no_decrease_counter = 0
max_no_decrease_steps = 10

# Generate data once
pypy_script_path = os.path.expanduser('dire_NLLc.py')
# Run the PyPy script and get the output filename
output_filename = run_pypy_script(pypy_script_path, n_samp, asif)

if output_filename:
    if os.path.exists(output_filename):
        tau_i = read_csv_to_torch(output_filename)
    else:
        print(f"CSV file not found: {output_filename}")
        raise RuntimeError("Data generation failed. Exiting.")
else:
    print("PyPy script execution failed.")
    raise RuntimeError("Data generation failed. Exiting.")

filtered_tau_0 = tau_i[(tau_i != 0.0)]
print("zero taus = ", len(filtered_tau_0))

min_tau = (torch.min(filtered_tau_0))
max_tau = (torch.max(filtered_tau_0))

print("min/max",min_tau,max_tau)
filtered_tau_i = tau_i[(tau_i >= min_tau) & (tau_i <= max_tau)]

CLL0 = torch_quad(wLL, min_tau, max_tau)
CLL = torch_quad(wLL, min_tau, max_tau, func_mul=analytics.R_L)/CLL0
CNLL = torch_quad(wLL, min_tau, max_tau, func_mul=analytics.R_NLLc)/CLL0
#CLL = torch_quad(wLL, min_tau, max_tau, func_mul=torch.log)#/CLL0

print("CLL0=",CLL0)
print("CLL=",CLL)
print("CNLL=",CNLL)
start_time = time.time()

for step in range(1000000):

    optimizer.zero_grad()
    loss_1 = torch.abs(integral_equation_1_direct(lambda_0, lambda_1, lambda_2, filtered_tau_i, n_samp))
    loss_2 = torch.abs(integral_equation_2_direct(lambda_0, lambda_1, lambda_2, filtered_tau_i, n_samp))
    loss = loss_2
    if torch.isnan(loss):
     continue
    loss.backward()
    optimizer.step()

    print(f"Step {step}, Loss: {loss.item()}, Lambda 0: {lambda_0.item()}, Lambda 1: {lambda_1.item()}, Lambda 2: {lambda_2.item()}")
    print("Time elapsed: {:.2f} seconds".format(time.time()-start_time))
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
        save_params(lambda_2.item(), asif, n_samp)
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
