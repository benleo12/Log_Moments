import math as m
import random as r
import nll
from vector import Vec4
from particle import Particle, CheckEvent
from qcd import AlphaS, NC, TR, CA, CF
import numpy as np

# build and run the generator
import sys, time, optparse
#from mpi4py import MPI
print("print here")
parser = optparse.OptionParser()
parser.add_option("-s","--seed",default=123456,dest="seed")
parser.add_option("-e","--events",default=10000,dest="events")
parser.add_option("-f","--file",default="alaric",dest="histo")
parser.add_option("-c","--collinear",default=3,dest="coll")
parser.add_option("-n","--nem",default=1000000,dest="nem")
parser.add_option("-N","--nmax",default=1000000,dest="nmax")
parser.add_option("-L","--lc",default=False,action="store_true",dest="lc")
parser.add_option("-a","--asmz",default='0.118',dest="asmz")
parser.add_option("-b","--beta",default='0',dest="beta")
parser.add_option("-A","--alphas",default=0.118,dest="alphas")
parser.add_option("-O","--order",default=1,dest="order")
parser.add_option("-M","--min",default=1,dest="min")
parser.add_option("-C","--cut",default=1,dest="cut")
parser.add_option("-R","--rcut",default='0',dest="rcut")
parser.add_option("-Q","--ecms",default='91.1876',dest="ecms")
parser.add_option("-F","--flat",default='[]',dest="flat")
parser.add_option("-S","--shower",default='A',dest="shower")
parser.add_option("-q","--quad",default=0,action="count",dest="quad")
parser.add_option("-K","--cluster",default=5,dest="cas")
parser.add_option("-l","--logfile",default="",dest="logfile")
(opts,args) = parser.parse_args()

opts.histo = opts.histo.format(**vars(opts))
if opts.logfile != "":
    sys.stdout = open(opts.logfile, 'w')

#comm = MPI.COMM_WORLD
#if comm.Get_rank() == 0:
#print('Running on {} ranks'.format(comm.Get_size()))
import config
config.quad_precision = int(opts.quad)
from mymath import *
#if comm.Get_rank() == 0:
print_math_settings()

from vector import Vec4, Rotation, LT
from particle import Particle, CheckEvent
from qcd import AlphaS, NC, TR, CA, CF
from analysis import SimplifiedAnalysis

import dire
import alaric

ecms = mn(opts.ecms)
lam = mn(opts.asmz)/mn(opts.alphas)
t0 = mypow(mn(opts.cut)/ecms**2,lam)*ecms**2
alphas = AlphaS(ecms,mn(opts.alphas),int(opts.order))

#if MPI.COMM_WORLD.Get_rank() == 0:
print("t_0 = {0}, log(Q^2/t_0) = {1}, \\alpha_s(t_0) = {2}". \
          format(t0,mylog(ecms**2/t0),alphas(t0)))
if opts.shower == 'A':
    shower = alaric.Shower(alphas,t0,int(opts.coll),mn(opts.beta),
                           mn(opts.rcut),eval(opts.flat),int(opts.nmax),opts.lc)
else:
    shower = dire.Shower(alphas,t0,int(opts.coll),mn(opts.beta),
                         mn(opts.rcut),eval(opts.flat),int(opts.nmax),opts.lc)
jetrat = SimplifiedAnalysis(-0.0033)

#rng.seed((comm.Get_rank()+1)*int(opts.seed))
rng.seed(int(opts.seed))
#start = time.perf_counter()
nevt, nout = int(float(opts.events)), 1
for i in range(1,nevt+1):
    event, weight = ( [
            Particle(-11,-Vec4(ecms/mn(2),mn(0),mn(0),ecms/mn(2)),[0,0],0),
            Particle(11,-Vec4(ecms/mn(2),mn(0),mn(0),-ecms/mn(2)),[0,0],1),
            Particle(1,Vec4(ecms/mn(2),mn(0),mn(0),ecms/mn(2)),[1,0],2,[3,0]),
            Particle(-1,Vec4(ecms/mn(2),mn(0),mn(0),-ecms/mn(2)),[0,1],3,[0,2])
        ], 1 )
    shower.Run(event,int(opts.nem))
    check = CheckEvent(event)
    if len(check): print('Error:',check[0],check[1])
    if i % nout == 0: #and comm.Get_rank() == 0:
        if opts.logfile != "":
            print('Event {n}\n'.format(n=i))
        else:
            sys.stdout.write('Event {n}\r'.format(n=i))
        sys.stdout.flush()
        if i/nout == 10: nout *= 10
    jetrat.Analyze(event,weight*shower.weight)
thrust_values = jetrat.Finalize()


import matplotlib.pyplot as plt
from scipy.integrate import quad
# Integration range
min_tau = 10**-35#10
max_tau = 1e-10#0.1


alphas = [ AlphaS(ecms,mn(opts.alphas),0),
           AlphaS(ecms,mn(opts.alphas),int(opts.order)) ]
analytics = nll.NLL(alphas,a=1,b=1,t=ecms**2)

# Define wLL and wNLL functions

# Define wLL and wNLL functions

def wLL(tau_values):
    # Check if tau_values is iterable (a list or numpy array), otherwise treat as a scalar
    try:
        iter(tau_values)
        is_scalar = False
    except TypeError:
        is_scalar = True
        tau_values = [tau_values]  # Treat the scalar as a single-element list

    results = []
    for tau in tau_values:
        partC = (analytics.Rp(tau)) / tau
        exponC = np.exp(-analytics.R_L(tau))
#        partC = ((analytics.RNNLLp(tau)-analytics.FpF(tau)))/tau
#        exponC = np.exp( - (analytics.R_SL(tau)-analytics.logF(tau)) )
        results.append(partC * exponC)

    if is_scalar:
        return results[0]  # Return the single result directly
    else:
        return np.array(results)  # Return an array of results


def wNLL(tau_values):
    # Check if tau_values is iterable (a list or numpy array), otherwise treat as a scalar
    try:
        iter(tau_values)
        is_scalar = False
    except TypeError:
        is_scalar = True
        tau_values = [tau_values]  # Treat the scalar as a single-element list

    results = []
    for tau in tau_values:
        partC = (analytics.Rp(tau) + (analytics.RNNLLp(tau)-analytics.FpF(tau)))/tau
        exponC = np.exp( -(analytics.R_SL(tau)-analytics.logF(tau)) -  analytics.R_L(tau))
        results.append(partC*exponC)
    return np.array(results)

    if is_scalar:
        return results[0]  # Return the single result directly
    else:
        return np.array(results)  # Return an array of results

# Numerical integration
CLL0, _ = quad(lambda t: wLL(t), min_tau, max_tau)
CLL1, _ = quad(lambda t: wLL(t) * np.log(t), min_tau, max_tau)
CLL2, _ = quad(lambda t: wLL(t)  * (analytics.R_L(t)), min_tau, max_tau)

CNLL0, _ = quad(lambda t: wNLL(t), min_tau, max_tau)
CNLL1, _ = quad(lambda t: wNLL(t) * ((analytics.R_SL(t)-analytics.logF(t))), min_tau, max_tau)
CNLL2, _ = quad(lambda t: wNLL(t) * (analytics.R_L(t)), min_tau, max_tau)

print("CLL2 = ",CLL2)

#lambda_0 = 0.281920760869
lambda_0 = 0
#lambda_1 = 0.31372693181037903
#lambda_2 = 0.25469261407852173

#lambda_0 = 0
#lambda_1 = 0.44
lambda_2 = 0.0

lambda_1 = 0.0
lambda_2 = 0.1319749653339386
lambda_2 = 2.3842785358428955

def weight(tau_i):
#return  np.exp(lambda_0 - lambda_1 * np.log(t) - lambda_2 * np.log(t)**2)
#print("tau=",tau_i)
    if tau_i<min_tau:
       return 0
    Rpt = analytics.Rp(tau_i)
    logFt = analytics.logF(tau_i)
    FpFt = analytics.FpF(tau_i)
    RNNLLpt = analytics.RNNLLp(tau_i)
    part = (CLL2)/((CLL2-lambda_2))
    expon = np.exp(-lambda_2*analytics.R_L(tau_i))
    return expon*part

# Generate tau values, avoiding 0 to prevent division by zero or log(0)
tau_values = np.logspace(np.log10(min_tau), -0.001, 100000)  # Upper limit set to 0.5 for better visualization

# Calculate theoretical probability densities
pLL_values = wLL(tau_values)
pNLL_values = wNLL(tau_values)
print("pLL_values=",pLL_values)
# Now plot the theoretical distribution
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1,1,1)
ax.set_yscale('log')
ax.plot(np.log10(tau_values), np.log(10)*tau_values*pLL_values, label='LL Theoretical Thrust Distribution', color='blue')
ax.plot(np.log10(tau_values), np.log(10)*tau_values*pNLL_values, label='NLL Theoretical Thrust Distribution', color='green')
# Overlay with actual thrust distribution
# Assuming 'thrust_values' is a list of thrust values from the previous operations
linbins = np.linspace(np.log10(min_tau),0.0,num=100)
#print("logbins=",logbins)

# Calculating weights for each thrust value
weights = [weight(tau) for tau in thrust_values]

# Weighted histogram
print("log10vals=",np.log10(thrust_values))
zeros=np.ones(len(thrust_values))*min_tau

plt.hist(np.log10(np.maximum(thrust_values,zeros)), bins=linbins, weights=weights, alpha=0.5, density=True, label='Weighted Simulated Distribution', color='black')
#    return coeff * ((-4 * np.log(t) + 3) / t) * np.exp(- coeff * (2 * np.log(t)**2 - 3 * np.log(t)))

# Unweighted histogram
plt.hist(np.log10(thrust_values), bins=linbins, alpha=0.5, density=True, label='Unweighted Simulated Distribution', color='red')


plt.title('Thrust Distribution with Theoretical Overlay')

plt.xlabel('Thrust, $\\tau$')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
