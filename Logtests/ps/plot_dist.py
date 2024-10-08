import math as m
import random as r
import nll
from vector import Vec4
from particle import Particle, CheckEvent
from qcd import AlphaS, NC, TR, CA, CF
import numpy as np

# build and run the generator
import sys, time, optparse
parser = optparse.OptionParser()
parser.add_option("-s","--seed",default=123456,dest="seed")
parser.add_option("-e","--events",default=10000,dest="events")
parser.add_option("-f","--file",default="alaric",dest="histo")
parser.add_option("-c","--collinear",default=3,dest="coll")
parser.add_option("-n","--nem",default=1,dest="nem")
parser.add_option("-N","--nmax",default=1000000,dest="nmax")
parser.add_option("-L","--lc",default=False,action="store_true",dest="lc")
parser.add_option("-a","--asmz",default='0.118',dest="asmz")
parser.add_option("-b","--beta",default='1',dest="beta")
parser.add_option("-A","--alphas",default=0.118,dest="alphas")
parser.add_option("-O","--order",default=1,dest="order")
parser.add_option("-M","--min",default=1,dest="min")
parser.add_option("-C","--cut",default=1,dest="cut")
parser.add_option("-R","--rcut",default='0',dest="rcut")
parser.add_option("-Q","--ecms",default='91.1876',dest="ecms")
parser.add_option("-S","--shower",default='D',dest="shower")
parser.add_option("-q","--quad",default=0,action="count",dest="quad")
parser.add_option("-k","--cluster",default=5,dest="cas")
parser.add_option("-l","--logfile",default="",dest="logfile")
parser.add_option("-x","--piece",default="all",dest="piece")
parser.add_option("-K","--Kfactor",default=1,dest="Kfac")
parser.add_option("-B","--Blfactor",default=1,dest="Blfac")
(opts,args) = parser.parse_args()

opts.histo = opts.histo.format(**vars(opts))
if opts.logfile != "":
    sys.stdout = open(opts.logfile, 'w')

import config
config.quad_precision = int(opts.quad)
config.Kfac = float(opts.Kfac)
config.Blfac = float(opts.Blfac)
from mymath import *
print_math_settings()

from vector import Vec4, Rotation, LT
from particle import Particle, CheckEvent
from qcd import AlphaS, NC, TR, CA, CF
from analysis import SimplifiedAnalysis
import os, subprocess, csv

import dire
import alaric

ecms = mn(opts.ecms)
lam = mn(opts.asmz)/mn(opts.alphas)
t0 = mypow(mn(opts.cut)/ecms**2,lam)*ecms**2

if opts.piece == 'll':
    K=0
    opts.nem=1
    opts.coll=0
    opts.order=0
elif opts.piece == 'nllc':
    opts.nem=1
    opts.coll=0
    opts.order=1
elif opts.piece == 'nll1':
    opts.nem=1
    opts.order=1

alphas = [ AlphaS(ecms,mn(opts.alphas),int(opts.order)),
           AlphaS(ecms,mn(opts.alphas),0) ]

def run_pypy_script(pypy_script_path, asif, n_samp, piece, kfac, blfac):
    flags = ['-e', str(n_samp), '-A', str(asif), '-b', '1', '-C', '0.1','-x',str(piece),'-K',str(kfac),'-B',str(blfac)]
    filename = [ f"thrust_e{n_samp}_A{asif}_{opts.piece}_K{kfac}_B{blfac}.csv",
                 f"weight_e{n_samp}_A{asif}_{opts.piece}_K{kfac}_B{blfac}.csv" ]
    print('Input files:',filename)
    if os.path.exists(filename[0]):
        with open(filename[0], 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            tau_values = [float(row[0]) for row in csv_reader]
        with open(filename[1], 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            wgt_values = [float(row[0]) for row in csv_reader]
        return tau_values, wgt_values
    subprocess.check_call(['pypy', pypy_script_path] + flags)
    os.rename("thrust_values.csv", filename[0])
    os.rename("weight_values.csv", filename[1])
    with open(filename[0], 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        tau_values = [float(row[0]) for row in csv_reader]
    with open(filename[1], 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        wgt_values = [float(row[0]) for row in csv_reader]
    return tau_values, wgt_values

pypy_script_path = os.path.expanduser('dire.py')
thrust_values, weight_values = run_pypy_script(pypy_script_path,
    opts.alphas, opts.events, opts.piece, opts.Kfac, opts.Blfac)

import matplotlib.pyplot as plt
from scipy.integrate import quad
# Integration range
min_tau = min([ i if i>0. else 1. for i in thrust_values ])
max_tau = 0.999

analytics = nll.NLL(alphas,a=1,b=1,t=ecms**2,piece=opts.piece)

# Define wLL and wNLL functions

def wLL(tau_values):
    try:
        iter(tau_values)
        is_scalar = False
    except TypeError:
        is_scalar = True
        tau_values = [tau_values]
    results = []
    for tau in tau_values:
        partC = (analytics.R_LLp(tau)) / tau
        exponC = np.exp(-analytics.R_LL(tau))
        results.append(partC * exponC)
    if is_scalar: return results[0]
    return np.array(results)

def wNLL(tau_values):
    try:
        iter(tau_values)
        is_scalar = False
    except TypeError:
        is_scalar = True
        tau_values = [tau_values]
    results = []
    for tau in tau_values:
        partC = (analytics.R_LLp(tau)+analytics.R_NLLp(tau)+analytics.FpF(tau))/tau
        exponC = np.exp(-analytics.R_LL(tau)-analytics.R_NLL(tau)-analytics.logF(tau))
        results.append(partC*exponC)
    if is_scalar: return results[0]
    return np.array(results)

# Numerical integration
CLL0, _ = quad(lambda t: wNLL(t), min_tau, max_tau)
CNLL, _ = quad(lambda t: wNLL(t) * ((analytics.R_NLL(t)-analytics.logF(t))), min_tau, max_tau)
CLL, _ = quad(lambda t: wNLL(t) * (analytics.R_LL(t)), min_tau, max_tau)

lambda_0 = 0.0
lambda_1, lambda_2, npm1, npm2, nps1, nps2, npn1, npn2 = 0.008303365196098627, 0.028592561718013335, 0.30143915006392746, 0.300065001025007, 0.07008604234713191, 0.23260547429703668, 0.07165096510758996, 0.13307553088856458

def weight(tau_i,w_i):
    RLLp = analytics.R_LLp(tau_i)
    RNLLp = analytics.R_NLLp(tau_i)
    partn = (CLL * RLLp + CNLL * RNLLp) 
    partd = ((CLL - lambda_1) * RLLp + (CNLL - lambda_2) * RNLLp)
    expon = np.exp( - lambda_1*(analytics.R_NLL(tau_i)) -  lambda_2*analytics.R_LL(tau_i))
    sudth = np.exp( -(analytics.R_NLL(tau_i)) -  analytics.R_LL(tau_i))
    gauss1 = np.exp(-0.5* (m.log(tau_i)-m.log(npm1)) ** 2 /nps1**2)*npn1*tau_i 
    gauss2 = np.exp(-0.5* (m.log(tau_i)-m.log(npm2)) ** 2 /nps2**2)*npn2*tau_i
    return w_i*expon*(partn + (gauss1-gauss2)/sudth)/partd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
# ... [other imports and your existing code] ...

# Function to export theoretical distributions
def export_theoretical_data(tau_values, pLL_values, pNLL_values, filename):
    theoretical_data = {
        'Tau_log10': np.log10(tau_values),  # Log10 of tau values
        'LL_Theoretical': np.log(10) * tau_values * pLL_values,
        'NLL_Theoretical': np.log(10) * tau_values * pNLL_values
    }
    df_theoretical = pd.DataFrame(theoretical_data)
    df_theoretical.to_csv(filename, index=False)
    print(f"Theoretical distributions exported to {filename}")

# Function to export simulated thrust data
def export_thrust_data(thrust_values, weights, filename):
    thrust_data = {
        'Thrust_log10': np.log10(np.maximum(thrust_values, 1e-10)),  # Avoid log(0)
        'Weights': weights
    }
    df_thrust = pd.DataFrame(thrust_data)
    df_thrust.to_csv(filename, index=False)
    print(f"Simulated thrust data exported to {filename}")

# Function to export interpolated histogram data
def export_interpolated_histogram(bin_centers, interp_values, filename):
    histogram_data = {
        'Bin_Center_log10': bin_centers,  # Log10 of bin centers
        'Interpolated_Values': interp_values
    }
    df_histogram = pd.DataFrame(histogram_data)
    df_histogram.to_csv(filename, index=False)
    print(f"Interpolated histogram data exported to {filename}")

# ... [rest of your existing code] ...

# After calculating theoretical distributions
# Generate tau values, avoiding 0 to prevent division by zero or log(0)
tau_values = np.logspace(mylog10(min_tau), -0.001, 10000)  # Upper limit set to 0.5 for better visualization

# Calculate theoretical probability densities
pLL_values = wLL(tau_values)
pNLL_values = wNLL(tau_values)

# Export theoretical distributions to CSV
export_theoretical_data(tau_values, pLL_values, pNLL_values, 'theoretical_distribution.csv')

# Calculating weights for each thrust value
weights = [weight(thrust,weigh) for thrust, weigh in zip(thrust_values,weight_values)]

# After calculating thrust data
# Export thrust data to CSV
export_thrust_data(thrust_values, weights, 'simulated_thrust.csv')

# After calculating interpolated histogram data
# Bin thrust values for interpolation
linbins = np.linspace(mylog10(min_tau), -0.001, num=100)
hist_vals, bin_edges = np.histogram(np.log10(np.maximum(thrust_values, 1e-10)), bins=linbins, density=True)

# Find the bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Perform cubic spline interpolation
interp_function = interp1d(bin_centers, hist_vals, kind='cubic', fill_value='extrapolate')
interp_values = interp_function(bin_centers)

# Export interpolated histogram data to CSV
export_interpolated_histogram(bin_centers, interp_values, 'interpolated_histogram.csv')

# Now plot the theoretical distribution and interpolation
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)

# Plot LL and NLL theoretical distributions
ax.plot(np.log10(tau_values), np.log(10)*tau_values*pLL_values, label='LL Theoretical Thrust Distribution', color='blue')
ax.plot(np.log10(tau_values), np.log(10)*tau_values*pNLL_values, label='NLL Theoretical Thrust Distribution', color='green')

# Weighted histogram
zeros = np.ones(len(thrust_values)) * 1e-10  # Avoid log(0)
plt.hist(np.log10(thrust_values), bins=linbins, weights=np.array(weights), alpha=0.5, density=True, label='Weighted Simulated Distribution', color='black')

# Unweighted histogram
plt.hist(np.log10(thrust_values), bins=linbins, weights=np.array(weight_values), alpha=0.5, density=True, label='Unweighted Simulated Distribution', color='red')

# Plot interpolated curve
# ax.plot(bin_centers, interp_values, label='Interpolated Values (Cubic Spline)', color='orange', linestyle='--')

plt.title('Thrust Distribution with Theoretical Overlay and Interpolation')
plt.xlabel('Log10(Thrust), $\\log_{10}(\\tau)$')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
