import argparse
print("Script started")
# other imports and code follow


parser = argparse.ArgumentParser(description='Process events and analyze thrust.')
parser.add_argument('--e', type=int, default=20000, help='Number of samples to process')
parser.add_argument('--n_step', type=int, default=100, help='Number of steps in the process')
parser.add_argument('--step_frac', type=float, default=1, help='Fractional step increase per iteration')
parser.add_argument('--samp_fac', type=float, default=1, help='Sample factor for scaling')
parser.add_argument('--asif', type=float, default=0.02, help='alphas limit')
parser.add_argument('--piece', default='ll', help='piece to fit')
parser.add_argument('--lam1', type=float, default='0')
parser.add_argument('--lam2', type=float, default='0')
parser.add_argument('--nbins', type=int, default='16')
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


# Define a function to save parameters to a file
def save_params(lambda_1,lambda_2, asif):
    filename = f"params_{args.piece}_{n_samp}.txt"
    with open(filename, 'w') as f:
        f.write(f"lambda_1: {lambda_1}\n")
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
torch.set_num_threads(4)

# Set the number of threads used for inter-op parallelism
torch.set_num_interop_threads(4)

alphas = [ AlphaS(91.1876,asif,0), AlphaS(91.1876,asif,1) ]
analytics = nll_torch.NLL(alphas,a=1,b=1,t=91.1876**2,piece=args.piece)


# Define wLL and wNLL functions
def wLL(tau):
    partC = (analytics.R_LLp(tau)+analytics.R_NLLp(tau))/tau
    exponC = np.exp( -analytics.R_LL(tau)-analytics.R_NLL(tau) )
    return partC*exponC

def torch_quad(func, a, b, func_mul=None, func_mul2=None, num_steps=100000):
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

def run_pypy_script(pypy_script_path,asif,n_samp,piece):
    """Runs the PyPy script with specified flags that generates the CSV file."""
    # Define the additional flags as a list
    flags = ['-e',str(n_samp),'-A',str(asif),'-b','1','-C','1','-x',str(piece)]

    # Create the filename based on flags and ensure it includes "LL"
    filename = f"thrust_e{n_samp}_A{asif}_{args.piece}.csv"
    
    # Check if the file already exists
    if os.path.exists(filename):
        print(f"File {filename} already exists. Not overwriting.")
        return filename

    try:
        # Include the additional flags in the subprocess call
        subprocess.check_call(['pypy', pypy_script_path] + flags)
        
        # Check if the expected output file exists and rename it
        if os.path.exists("thrust_values.csv"):
            os.rename("thrust_values.csv", filename)
        else:
            print("Expected output file 'thrust_values.csv' not found.")
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
def prep_integral_equation_2_direct(lambda_0,lambda_1, lambda_2, tau_i, n_samp,printit=False):
    n_bins = args.nbins
    n_samp = int(list(tau_i.size())[0]/n_bins)*n_bins
    tau_i, indices = torch.sort(tau_i[:n_samp])
    bins = tau_i.reshape(n_bins,int(n_samp/n_bins))
    if printit:
        print('NBins',n_bins,tau_i.size())
    maxs, ids = torch.max(bins,1)
    mins, ids = torch.min(bins,1)
#    if printit:
#        print('Params',mins)
#        print('Params',maxs)
#        print('Params',cens)
    return [bins,mins,maxs,n_samp]

def integral_equation_2_direct(lambda_0,lambda_1, lambda_2, bins,printit=False):
    integral = 0
    RLLp = analytics.R_LLp(bins[0])
    RNLLp = analytics.R_NLLp(bins[0])
    part = (CLL*RLLp + CNLL*RNLLp)/((CLL-lambda_1)*RLLp + (CNLL-lambda_2)*RNLLp)
    expon = torch.exp( -lambda_1*analytics.R_LL(bins[0])-lambda_2*analytics.R_NLL(bins[0]) )
    vals = torch.sum(part*expon,1)/bins[3]
    anas  = np.exp( -analytics.R_LL(bins[2])-analytics.R_NLL(bins[2]) )
    anas -= np.exp( -analytics.R_LL(bins[1])-analytics.R_NLL(bins[1]) )
    integral = sum((vals-anas)**2)
    if printit:
        print('\\Chi^2 is',integral.item(),'at',lambda_1.item(),lambda_2.item())
    return integral

# Initialize the Lagrange multipliers
lambda_0 = torch.tensor([0.0], requires_grad=True)
lambda_1 = torch.tensor([args.lam1], requires_grad=True)
lambda_2 = torch.tensor([args.lam2], requires_grad=True)

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
pypy_script_path = os.path.expanduser('dire.py')
# Run the PyPy script and get the output filename
output_filename = run_pypy_script(pypy_script_path, asif, n_samp, args.piece)

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
print("Nonzero tau values: ", len(filtered_tau_0))

# Now you can use min_tau and peak_tau in your further calculations
min_tau = torch.min(filtered_tau_0)
max_tau = torch.max(filtered_tau_0)

print("Tau min/max: ",min_tau.item(),max_tau.item())

filtered_tau_i = tau_i[(tau_i >= min_tau) & (tau_i <= max_tau)]

CN   = torch_quad(wLL, min_tau, max_tau)
CLL  = torch_quad(wLL, min_tau, max_tau, func_mul=analytics.R_LL)
CNLL = torch_quad(wLL, min_tau, max_tau, func_mul=analytics.R_NLL)

print("CN   =",CN)
print("CLL  =",CLL)
print("CNLL =",CNLL)

# Define the optimizer
defrate = 0.01
if args.piece == 'll':
    optimizer = optim.Adam([{'params': lambda_1, 'lr': defrate}])
if args.piece == 'nllc' or args.piece == 'nll1':
    optimizer = optim.Adam([{'params': lambda_2, 'lr': defrate}])

bins = prep_integral_equation_2_direct(lambda_0, lambda_1, lambda_2, filtered_tau_i, n_samp,True)

integral_equation_2_direct(lambda_0, lambda_1, lambda_2, bins, True)

for step in range(1000000):

    #print(f"Running optimization step {step+1}")

    optimizer.zero_grad()
    loss = torch.abs(integral_equation_2_direct(lambda_0, lambda_1, lambda_2, bins))

    if torch.isnan(loss):
     continue
    loss.backward()
    optimizer.step()
#    with torch.no_grad():
#        lambda_2 = lambda_2.clamp(min=0)

    print("Step {}, Loss: {}, Lambda 0: {}, Lambda 1: {}, Lambda 2: {}\r".format(step,loss.item(),lambda_0.item(),lambda_1.item(),lambda_2.item()), end='', flush=True)

    #if step >0 and step % (n_step+5) == 0:
    #print(f"Step {step}, Loss: {loss.item()}, Lambda 0: {lambda_0.item()}, Lambda 1: {lambda_1.item()}, Lambda 2: {lambda_2.item()}")
    n_samp = int(n_samp*samp_fac)
    n_step = int(n_step/step_frac)
    # print("n_step = ", n_step)
    lambda_1_values.append(lambda_1.item())
    lambda_2_values.append(lambda_2.item())
    loss_values.append(loss.item())
    n_samp_values.append(n_samp)
    #if n_step <= 5:
    #    break

    # Save parameters if loss doesn't change
    if step > 0 and abs(loss.item() - loss_values[step-1]) < 1e-20:
        save_params(lambda_1.item(),lambda_2.item(), asif)
        break
    
#    if step > 0 and loss.item() < min_loss:
#        save_params(lambda_2.item(), asif)
#        break

print("")
print(f"Final Lambda 0: {lambda_0.item()}, Final Lambda 1: {lambda_1.item()}, Final Lambda 2: {lambda_2.item()}")
