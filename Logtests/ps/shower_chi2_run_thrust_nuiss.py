import argparse
print("Script started")
parser = argparse.ArgumentParser(description='Process events and analyze thrust.')
parser.add_argument('--e', type=int, default=20000, help='Number of samples to process')
parser.add_argument('--n_step', type=int, default=100, help='Number of steps in the process')
parser.add_argument('--step_frac', type=float, default=1, help='Fractional step increase per iteration')
parser.add_argument('--samp_fac', type=float, default=1, help='Sample factor for scaling')
parser.add_argument('--asif', type=float, default=0.02, help='alphas limit')
parser.add_argument('--piece', default='ll', help='piece to fit')
parser.add_argument('--min', type=float, default='0')
parser.add_argument('--max', type=float, default='1')
parser.add_argument('--lam1', type=float, default='-1')
parser.add_argument('--lam2', type=float, default='-1')
parser.add_argument('--nbins', type=int, default='16')
parser.add_argument('--nuisance_mu1', type=float, default=0, help='Mean of the first nuisance parameter')
parser.add_argument('--nuisance_sigma1', type=float, default=1, help='Standard deviation of the first nuisance parameter')
parser.add_argument('--nuisance_mu2', type=float, default=0, help='Mean of the second nuisance parameter')
parser.add_argument('--nuisance_sigma2', type=float, default=1, help='Standard deviation of the second nuisance parameter')
parser.add_argument('-K', type=float, default=1)
parser.add_argument('-B', type=float, default=1)
parser.add_argument('-F', type=float, default=1)
parser.add_argument('-C', type=float, default=1)
parser.add_argument('-s', type=int, default=1)
# Debugging: Verify this line is executed
print("Debug: Adding arguments complete")
import sys
print(sys.argv)

import sys

import config
config.use_torch = 1
# Parse arguments
args = parser.parse_args()
n_samp = args.e
n_step = args.n_step
step_frac = args.step_frac
samp_fac = args.samp_fac
asif = args.asif
config.Kfac = args.K
config.Blfac = args.B
config.Ffac = args.F

# Define a function to save parameters to a file
def save_params(lambda_1, lambda_2, npm1, npm2, nps1, nps2, npn1, npn2, asif):
    filename = f"params_{args.piece}_{n_samp}.txt"
    with open(filename, 'w') as f:
        f.write(f"lambda_1: {lambda_1}\n")
        f.write(f"lambda_2: {lambda_2}\n")
        f.write(f"npm1: {npm1}\n")
        f.write(f"npm2: {npm2}\n")
        f.write(f"nps1: {nps1}\n")
        f.write(f"nps2: {nps2}\n")
        f.write(f"npn1: {npn1}\n")
        f.write(f"npn2: {npn2}\n")
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
import nll as nll_torch
from qcd import AlphaS, NC, TR, CA, CF
from scipy.integrate import quad

torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# Set the number of threads used for intra-op parallelism
torch.set_num_threads(8)

# Set the number of threads used for inter-op parallelism
torch.set_num_interop_threads(8)

alphas = [AlphaS(91.1876, asif, 0), AlphaS(91.1876, asif, 1)]
analytics = nll_torch.NLL(alphas, a=1, b=1, t=91.1876**2, piece=args.piece)

# Define wLL and wNLL functions
def wLL(tau):
    partC = (analytics.R_LLp(tau) + analytics.R_NLLp(tau)) / tau
    exponC = torch.exp(-analytics.R_LL(tau) - analytics.R_NLL(tau))
    return partC * exponC

def torch_quad(func, a, b, func_mul=None, func_mul2=None, num_steps=100000):
    x = torch.logspace(torch.log10(a), torch.log10(b), steps=num_steps, dtype=torch.float64)
    dx = (x[1:] - x[:-1])
    y = (func(x[1:]) + func(x[:-1])) / 2.
    if func_mul is not None:
        y *= (func_mul(x[1:]) + func_mul(x[:-1])) / 2.
    if func_mul2 is not None:
        y *= (func_mul2(x[1:]) + func_mul2(x[:-1])) / 2.
    y_dx = y * dx
    integral = torch.sum(y_dx)
    return integral

def run_pypy_script(pypy_script_path, asif, n_samp, piece):
    """Runs the PyPy script with specified flags that generates the CSV file."""
    # Define the additional flags as a list
    flags = ['-e', str(n_samp), '-A', str(asif), '-b', '1', '-C', '1', '-x', str(piece), '-K', str(args.K), '-B', str(args.B), '-s', str(args.s)]

    # Create the filename based on flags and ensure it includes "LL"
    filename = [f"thrust_e{n_samp}_A{asif}_{args.piece}_K{args.K}_B{args.B}_seed{args.s}.csv",
                f"weight_e{n_samp}_A{asif}_{args.piece}_K{args.K}_B{args.B}_seed{args.s}.csv"]

    # Check if the file already exists
    if os.path.exists(filename[0]):
        print(f"File {filename} already exists. Not overwriting.")
        return filename

    try:
        # Include the additional flags in the subprocess call
        subprocess.check_call(['pypy', pypy_script_path] + flags)

        # Check if the expected output file exists and rename it
        if os.path.exists("thrust_values.csv"):
            os.rename("thrust_values.csv", filename[0])
            os.rename("weight_values.csv", filename[1])
        else:
            print("Expected output file 'thrust_values.csv' not found.")
            return None

    except subprocess.CalledProcessError as e:
        print("An error occurred while running the PyPy script:", e)
        return None

    return filename

def read_csv_to_torch(csv_file_paths):
    """Reads the CSV file and converts it to a PyTorch tensor."""
    with open(csv_file_paths[0], 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        tau_values = [float(row[0]) for row in csv_reader]
    with open(csv_file_paths[1], 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        wgt_values = [float(row[0]) for row in csv_reader]
    return torch.stack([torch.tensor(tau_values), torch.tensor(wgt_values)], dim=1)
# Manual integration function using Riemann sum
def mc_integration(integrand, tau_values, n_samp):
    integral = torch.sum(integrand) / n_samp
    return integral

# Define the integral equations using the dataset directly
def prep_integral_equation_2_direct(lambda_0, lambda_1, lambda_2, tau_i, n_samp, printit=False):
    n_bins = args.nbins
    n_samp = int(list(tau_i.size())[0] / n_bins) * n_bins
    tau_i = tau_i[:n_samp]
    tau_i = tau_i[tau_i[:, 0].sort()[1]]
    bins = tau_i[:, 0].reshape(n_bins, int(n_samp / n_bins))
    wgts = tau_i[:, 1].reshape(n_bins, int(n_samp / n_bins))
    if printit:
        print('NBins', n_bins, tau_i.size())
    maxs, ids = torch.max(bins, 1)
    mins, ids = torch.min(bins, 1)
    #    if printit:
    #        print('Params',mins)
    #        print('Params',maxs)
    #        print('Params',cens)
    return [bins, wgts, mins, maxs, n_samp]


def integral_equation_2_direct(lambda_0, lambda_1, lambda_2,  bins, npm1, nps1, npn1, npm2, nps2, npn2, printit=False):
    total_weight = torch.sum(bins[1])
    wgts = bins[1]

        # Define wLL_func that depends on lambda_1 and lambda_2
    def wLL_func(tau):
        partC = (analytics.R_LLp(tau) + analytics.R_NLLp(tau)) / tau
        exponC = torch.exp(- analytics.R_LL(tau) - analytics.R_NLL(tau))
        return partC * exponC
    
    # Compute analytic moments via integration
    min_tau = bins[2].min()
    max_tau = bins[3].max()

    CN = torch_quad(wLL_func, min_tau, max_tau)
    CLL = torch_quad(wLL_func, min_tau, max_tau, func_mul=analytics.R_LL)
    CNLL = torch_quad(wLL_func, min_tau, max_tau, func_mul=analytics.R_NLL)

    analytic_R_LL = CLL / CN
    analytic_R_NLL = CNLL / CN

    RLLp = analytics.R_LLp(bins[0])
    RNLLp = analytics.R_NLLp(bins[0])
    partn = (CLL * RLLp + CNLL * RNLLp) 
    partd = ((CLL - lambda_1) * RLLp + (CNLL - lambda_2) * RNLLp)
    sudth = torch.exp(- analytics.R_LL(bins[0]) - analytics.R_NLL(bins[0]))
    expon = torch.exp(-lambda_1 * analytics.R_LL(bins[0]) - lambda_2 * analytics.R_NLL(bins[0]))
    gauss1 = torch.exp(-0.5*(torch.log(bins[0])-torch.log(npm1)) ** 2 /nps1**2)*npn1 * bins[0]
    gauss2 = torch.exp(-0.5*(torch.log(bins[0])-torch.log(npm2)) ** 2 /nps2**2)*npn2 * bins[0]
    vals = wgts*expon*(partn + (gauss1-gauss2)/sudth)/partd
    

    # Compute data moments
    data_R_LL = torch.sum(vals*analytics.R_LL(bins[0])) / bins[4]
    data_R_NLL = torch.sum(vals*analytics.R_NLL(bins[0])) / bins[4]

    loss = 0

    # Compute loss
    loss = (data_R_LL - analytic_R_LL) ** 2 + (data_R_NLL - analytic_R_NLL) ** 2

    if printit:
        print('Loss is', loss.item(), 'at', lambda_1.item(), lambda_2.item())

    return loss


# Generate data once
pypy_script_path = os.path.expanduser('dire.py')
# Run the PyPy script and get the output filename
output_filename = run_pypy_script(pypy_script_path, asif, n_samp, args.piece)

if output_filename[0]:
    if os.path.exists(output_filename[0]):
        tau_i = read_csv_to_torch(output_filename)
    else:
        print(f"CSV file not found: {output_filename}")
        raise RuntimeError("Data generation failed. Exiting.")
else:
    print("PyPy script execution failed.")
    raise RuntimeError("Data generation failed. Exiting.")

min_tau = tau_i.min(dim=0).values[0]
max_tau = tau_i.max(dim=0).values[0]

tau_i = tau_i[ tau_i[:,0] > min_tau ]
tau_i = tau_i[ tau_i[:,0] < max_tau ]

print("Tau min/max:", min_tau.item(), max_tau.item())

# Initialize the Lagrange multipliers
lambda_0 = torch.tensor([0.0], requires_grad=True)
lambda_1 = torch.tensor([max(args.lam1,0.0)], requires_grad=True)
lambda_2 = torch.tensor([max(args.lam2,0.0)], requires_grad=True)

# Initialize the nuisance parameters
npm1 = torch.tensor([0.8*max_tau], requires_grad=True)
npm2 = torch.tensor([max_tau], requires_grad=True)
nps1 = torch.tensor([0.01], requires_grad=True)
nps2 = torch.tensor([0.02], requires_grad=True)
npn1 = torch.tensor([0.01], requires_grad=True)
npn2 = torch.tensor([0.02], requires_grad=True)

# Optimization loop
# Lists to collect data
lambda_1_values = []
lambda_2_values = []
loss_values = []
n_samp_values = []

CN = torch_quad(wLL, min_tau, max_tau)
CLL = torch_quad(wLL, min_tau, max_tau, func_mul=analytics.R_LL)
CNLL = torch_quad(wLL, min_tau, max_tau, func_mul=analytics.R_NLL)

print("CN   =", CN)
print("CLL  =", CLL)
print("CNLL =", CNLL)

# Define the optimizer, including nuisance parameters
defrate = 0.01
parms= [
    {'params': npm1, 'lr': defrate},
    {'params': npm2, 'lr': defrate},
    {'params': nps1, 'lr': defrate},
    {'params': nps2, 'lr': defrate},
    {'params': npn1, 'lr': defrate},
    {'params': npn2, 'lr': defrate}
]
if args.piece == 'll':
    parms.append({'params': lambda_1, 'lr': defrate})
if args.piece == 'nllc' or args.piece == 'nll1':
    parms.append({'params': lambda_1, 'lr': defrate})
    parms.append({'params': lambda_2, 'lr': defrate})
optimizer = optim.Adam(parms)

bins = prep_integral_equation_2_direct(lambda_0, lambda_1, lambda_2, tau_i, n_samp, True)

integral_equation_2_direct(lambda_0, lambda_1, lambda_2, bins, npm1, nps1, npn1, npm2, nps2, npn2, True)

for step in range(1000000):

    optimizer.zero_grad()
    loss = torch.abs(integral_equation_2_direct(lambda_0, lambda_1, lambda_2, bins, npm1, nps1, npn1, npm2, nps2, npn2))

    if torch.isnan(loss):
        continue
    loss.backward()
    optimizer.step()

    #with torch.no_grad():
    #    npm1.clamp_(0, 1)
    #    npm2.clamp_(0, 1)
        #npn1.clamp_(0, 5)
        #npn2.clamp_(0, 5)

    print("\rStep {}: Loss={}; lambda_1, lambda_2, npm1, npm2, nps1, nps2, npn1, npn2 = {}, {}, {}, {}, {}, {}, {}, {}".format(
        step, loss.item(), lambda_1.item(), lambda_2.item(), npm1.item(), npm2.item(), nps1.item(), nps2.item(), npn1.item(),npn2.item()), end='', flush=True)

    n_samp = int(n_samp * samp_fac)
    n_step = int(n_step / step_frac)
    lambda_1_values.append(lambda_1.item())
    lambda_2_values.append(lambda_2.item())
    loss_values.append(loss.item())
    n_samp_values.append(n_samp)

    # Save parameters if loss doesn't change
    if step > 0 and abs(loss.item() - loss_values[step - 1]) < 1e-10:
        save_params(lambda_1.item(), lambda_2.item(), npm1.item(), nps1.item(), npn1.item() , npm2.item(), nps2.item(), npn2.item(), asif)
        break

print("")
print("Final: Loss={}; lambda_1, lambda_2, npm1, npm2, nps1, nps2, npn1, npn2 = {}, {}, {}, {}, {}, {}, {}, {}\n".format(
        loss.item(), lambda_1.item(), lambda_2.item(), npm1.item(), npm2.item(), nps1.item(), nps2.item(), npn1.item(),npn2.item()), end='', flush=True)
