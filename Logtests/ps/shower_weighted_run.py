#import math as m
import random as r
import nll
from vector import Vec4, Rotation, LT
from particle import Particle, CheckEvent
from qcd import AlphaS, NC, TR, CA, CF
import numpy as np
from mymath import *

K = (67./18.-pow(m.pi,2)/6.)*CA-10./9.*TR*5
#K = 0.0 
class Kernel:

    def __init__(self,flavs,Ca,alpha,t0):
        self.flavs = flavs
        self.Ca = Ca
        self.alphamax=alpha[1](t0)
        self.alpha=alpha

class Soft (Kernel):

    def Value(self,z,k2,t):
        asrat = self.alpha[1](t)/self.alpha[0](t)
        return self.Ca*(2*(-z[1])/(z[1]*z[1]+k2))*(1+asrat*self.alpha[0](t)/(2*m.pi)*K)
    def Estimate(self,z,k02):
        return self.Ca*2*(-z[1])/(z[1]*z[1]+k02)*(1+self.alphamax/(2*m.pi)*K)
    def Integral(self,k02):
        return self.Ca*mylog(1+1/k02)*(1+self.alphamax/(2*m.pi)*K)
    def GenerateZ(self,k02):
        return [mn(1),-mysqrt(k02*(mypow(1+1/k02,mn(rng.random()))-1))]

class Cqq (Kernel):

    def Value(self,z,k2,t):
        return self.Ca*(-2+1-z[1])
    def Estimate(self,z,k02):
        return self.Ca
    def Integral(self,k02):
        return self.Ca
    def GenerateZ(self,k02):
        return [mn(0),mn(rng.random())]

class Cgg (Kernel):

    def Value(self,z,k2,t):
        return self.Ca*(-2+z[1]*(1-z[1]))
    def Estimate(self,z,k02):
        return self.Ca
    def Integral(self,k02):
        return self.Ca
    def GenerateZ(self,k02):
        return [mn(0),mn(rng.random())]

class Cgq (Kernel):

    def Value(self,z,k2,t):
        return TR/2*(1-2*z[1]*(1-z[1]))
    def Estimate(self,z,k02):
        return TR/2
    def Integral(self,k02):
        return 5*TR/2
    def GenerateZ(self,k02):
        fl = rng.randint(1,5)
        self.flavs[1] = fl
        self.flavs[2] = -fl
        return [mn(0),mn(rng.random())]

class Shower:

    def __init__(self,alpha,t0,coll,beta,rt0,flat,nmax,lc):
        self.nmax = nmax
        self.t0 = t0
        self.rt0 = rt0
        self.beta = beta
        if len(flat)!=2: self.flat = False
        else:
            self.flat = True
            self.lmin = flat[0]
            self.lmax = flat[1]
        self.alpha = alpha[0]
        self.alphamax = alpha[0](self.t0)
        self.amode = 0 if self.alpha.order == -1 else 1
        if self.amode != 0:
            self.alphamax = (2*m.pi)/self.alpha.beta0(5)
        self.kernels = {}
        for fl in [-5,-4,-3,-2,-1,1,2,3,4,5]:
            self.kernels[fl] = [ Soft([fl,fl,21],CA/2 if lc else CF,alpha,self.t0) ]
        self.kernels[21] = [ Soft([21,21,21],CA/2,alpha,self.t0) ]
        if coll & 1:
            for fl in [-5,-4,-3,-2,-1,1,2,3,4,5]:
                self.kernels[fl].append( Cqq([fl,fl,21],CA/2 if lc else CF,alpha,self.t0) )
        if coll & 2:
            self.kernels[21].append( Cgg([21,21,21],CA/2,alpha,self.t0) )
            self.kernels[21].append( Cgq([21,0,0],0,alpha,self.t0) )
        self.Ca = CA/2 if lc else CF
        self.Bl = -3./4. if coll&1 else -1.

    def dsigma(self,v):
        if self.alpha.order >= 0:
            print('Order \\alpha not supported')
            exit(1)
        cx = mylog(v)
        J = 1/(1+self.beta)
        aS = self.alpha(self.Q2,5)
        return myexp(-J*aS*self.Ca/(m.pi)* \
                     (cx**2/mn(2)-2*self.Bl*cx+mn(3)/2-v))* \
            (-J*aS*self.Ca/(m.pi)*(cx-2*self.Bl-v))

    def MakeKinematics(self,x,y,phi,pijt,pkt):
        Q2 = 2*pijt.SmallMLDP(pkt)
        z, omz = x[0]+x[1], 1-x[1] if x[0] == 0 else -x[1]
        if z < 0 or omz < 0: return []
        rkt = mysqrt(Q2*y*z*omz)
        kt1 = pijt.Cross(Rotation(pijt,pkt).y)
        if kt1.P2() == 0: return []
        kt1 *= rkt*mycos(phi)/kt1.P()
        uijk = pijt*pkt[0]-pkt*pijt[0]
        if uijk[3] == 0 and \
           pijt[1] == -pkt[1] and \
           pijt[2] == -pkt[2]: return []
        kt2 = kt1.Cross(uijk)
        kt2[0] = pijt*kt1.Cross(pkt)
        if kt2.M2() == 0: return []
        kt2 *= rkt*mysin(phi)/mysqrt(abs(kt2.M2()))
        pi = z*pijt + omz*y*pkt + kt1 + kt2
        pj = omz*pijt + z*y*pkt - kt1 - kt2
        pk = (1-y)*pkt
        if pi[0] <= 0 or pj[0] <= 0: return []
        return [pi,pj,pk]

    def MakeColors(self,flavs,colij,colk):
        self.c += 1
        if flavs[0] != 21:
            if flavs[0] > 0:
                return [ [self.c,0], [colij[0],self.c] ]
            else:
                return [ [0,self.c], [self.c,colij[1]] ]
        else:
            if flavs[1] == 21:
                if colij[0] == colk[1]:
                    if colij[1] == colk[0] and rng.random()>0.5:
                        return [ [colij[0],self.c], [self.c,colij[1]] ]
                    return [ [self.c,colij[1]], [colij[0],self.c] ]
                else:
                    return [ [colij[0],self.c], [self.c,colij[1]] ]
            else:
                if flavs[1] > 0:
                    return [ [colij[0],0], [0,colij[1]] ]
                else:
                    return [ [0,colij[1]], [colij[0],0] ]

    def UpdateWeights(self,split,event):
        gsum, osum = mn(0), self.gs[split.id][1]
        self.gs[split.id] = [self.gs[split.id][0],mn(0)]
        for cp in split.cps:
            if cp == 0: continue
            spect = event[cp]
            for i,sf in enumerate(self.kernels[split.pid]):
                m2 = 2*split.mom.SmallMLDP(spect.mom)
                if m2 <= 0: g = mn(0)
                else:
                    g = self.alphamax/(2*m.pi)*sf.Integral(self.ct0/m2)
                    if g <= 0: g = mn(0)
                self.gs[split.id].append([gsum+g,split.id,spect.id,i,m2])
                gsum += g
        self.gs[split.id][1] = gsum
        for g in self.gs[split.id:]: g[0] += gsum-osum

    def GenerateZ(self,event,momsum,s,t):
        z = s[2].GenerateZ(self.ct0/s[3])
        omz = 1-z[1] if z[0] == 0 else -z[1]
        if omz == 0: return False
        Q2, sijt = momsum.M2(), 2*s[0].mom.SmallMLDP(s[1].mom)
        sit, sjt = 2*momsum*s[0].mom, 2*momsum*s[1].mom
        if sijt <= 0 or sit >= 0 or sjt >= 0: return False
        v, rho = mysqrt(t/Q2), mypow(sit*sjt/(Q2*sijt),self.beta/2)
        Q, a, b = mysqrt(sijt/Q2*sit/sjt), 1, self.beta
        kt = mysqrt(Q2)*mypow(rho*v,a/(a+b))*mypow(Q*omz,b/(a+b))
        y = kt**2/s[3]/omz
        x = [ z[0], (z[1]-y)/(1-y) if z[0] == 0 else z[1]/(1-y) ]
        if y <= 0 or y >= 1: return False
        if self.amode == 0:
            w = self.alpha(kt**2,5)/self.alphamax
        else:
            asref = self.alpha.asa(t,5)
            if asref>0: w = self.alpha(kt**2,5)/asref
            else: w = 0
        w *= s[2].Value(z,kt**2/s[3],kt**2)/s[2].Estimate(z,self.ct0/s[3])/(1-y)
        w *= 1/(1+self.beta)
        if w > 1.0:
            print(w)
        #if w <= rng.random(): return False
        phi = 2*m.pi*rng.random()
        moms = self.MakeKinematics(x,y,phi,s[0].mom,s[1].mom)
        if moms == []: return False
        cps = []
        for cp in s[0].cps:
            if cp != 0: cps.append(cp)
        cols = self.MakeColors(s[2].flavs,s[0].col,s[1].col)
        event.append(Particle(s[2].flavs[2],moms[1],cols[1],len(event),[0,0]))
        for c in [0,1]:
            if cols[1][c] != 0 and cols[1][c] == s[0].col[c]:
                event[-1].cps[c] = s[0].cps[c]
                event[s[0].cps[c]].cps[1-c] = event[-1].id
                s[0].cps[c] = 0
        s[0].Set(s[2].flavs[1],moms[0],cols[0])
        for c in [0,1]:
            if cols[0][c] != 0 and cols[0][c] == cols[1][1-c]:
                event[-1].cps[1-c] = s[0].id
                s[0].cps[c] = event[-1].id
        s[1].mom = moms[2]
        s[2] = event[-1]
        self.UpdateWeights(s[0],event)
        for cp in cps: self.UpdateWeights(event[cp],event)
        self.gs.append([self.gs[-1][0],mn(0)])
        self.UpdateWeights(s[2],event)
        return True

    def SelectSplit(self,event,rn1,rn2):
        l, r = 0, len(self.gs)-1
        c = int((l+r)/2)
        a, d = self.gs[c][0], rn1*self.gs[-1][0]
        while r-l > 1:
            if d < a: r = c
            else: l = c
            c = int((l+r)/2)
            a = self.gs[c][0]
        if d < self.gs[l][0]: r = l
        k = r
        l, r = 2, len(self.gs[k])-1
        c = int((l+r)/2)
        a, d = self.gs[k][c][0], rn2*self.gs[k][1]
        while r-l > 1:
            if d < a: r = c
            else: l = c
            c = int((l+r)/2)
            a = self.gs[k][c][0]
        if d < self.gs[k][l][0]: r = l
        if k != self.gs[k][r][1]: print('Error in integral table')
        sf = self.kernels[event[k].pid][self.gs[k][r][-2]]
        return [ event[k], event[self.gs[k][r][2]], sf, self.gs[k][r][-1] ]

    def GeneratePoint(self,event):
        momsum = event[0].mom+event[1].mom
        trials = 0
        while self.t > self.ct0:
            trials += 1
            if trials == self.nmax:
                print('Abort after',trials,'trials in rank',comm.Get_rank())
                self.t = self.ct0
                return
            t = self.ct0
            g = self.gs[-1][0]
            if self.amode == 0:
                tt = self.t*mypow(mn(rng.random()),1/g)
            else:
                l2 = self.alpha.l2a(5)
                tt = l2*mypow(self.t/l2,mypow(mn(rng.random()),1/g))
            if tt > t:
                t = tt
                s = self.SelectSplit(event,rng.random(),rng.random())
            if len(event) == 4 and self.flat:
                lmax = min(self.lmax,self.alpha(self.Q2,5)/2*mylog(mn(1)/16))
                lmin = self.lmin
                lam = lmax+(lmin-lmax)*rng.random()
                t = self.Q2*myexp(lam/(self.alpha(self.Q2,5)/2))
                self.weight *= (lmax-lmin)*2/self.alpha(self.Q2,5)
                self.t = t
                while True:
                    if self.GenerateZ(event,momsum,s,t):
                        self.weight *= self.dsigma(t/self.Q2)
                        return
                    s = self.SelectSplit(event,rng.random(),rng.random())
            self.t = t
            if self.t > self.ct0:
                if self.GenerateZ(event,momsum,s,t): return

    def Run(self,event,nem):
        em = 0
        self.c = 2
        self.weight = 1
        self.ct0 = self.t0
        self.Q2 = (event[0].mom+event[1].mom).M2()
        self.t = self.Q2
        self.gs = [[mn(0),mn(0)],[mn(0),mn(0)]]
        for split in event[2:]:
            self.gs.append([self.gs[-1][0],mn(0)])
            self.UpdateWeights(split,event)
        while self.t > self.ct0:
            if em >= nem: return
            self.GeneratePoint(event)
            return
            if em == 0 and self.rt0 != 0:
                self.ct0 = max(self.t0,self.t*self.rt0)
            em += 1

# build and run the generator
import sys, time, optparse
#from mpi4py import MPI
print("print here")
parser = optparse.OptionParser()
parser.add_option("-s","--seed",default=123456,dest="seed")
parser.add_option("-e","--events",default=100000,dest="events")
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
parser.add_option("-C","--cut",default=1.0,dest="cut")
parser.add_option("-R","--rcut",default='0',dest="rcut")
parser.add_option("-Q","--ecms",default='91.1876',dest="ecms")
parser.add_option("-F","--flat",default='[]',dest="flat")
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

ecms = mn(opts.ecms)
lam = mn(opts.asmz)/mn(opts.alphas)
t0 = mypow(mn(opts.cut)/ecms**2,lam)*ecms**2


opts.nem=1
opts.coll=0
opts.order=0


alphas = [ AlphaS(ecms,mn(opts.alphas),int(opts.order)),AlphaS(ecms,mn(opts.alphas),0) ]
#if MPI.COMM_WORLD.Get_rank() == 0:

print("t_0 = {0}, log(Q^2/t_0) = {1}, \\alpha_s(t_0) = {2} / {3}". \
          format(t0,mylog(ecms**2/t0),alphas[0](t0),alphas[1](t0)))
shower = Shower(alphas,t0,int(opts.coll),mn(opts.beta),
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

#print("thrust_values=",thrust_values)


import matplotlib.pyplot as plt
from scipy.integrate import quad
# Integration range
min_tau = 10**-3.0#10
max_tau = 0.999#0.1
#coeff = 2 * alpha_s / (3 * np.pi)

alphas = [ AlphaS(91.1876,0.118,0), AlphaS(91.1876,0.118,1) ]
analytics = nll.NLL(alphas,a=1,b=1,t=91.1876**2)


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
        partC = (analytics.Rp(tau) + (analytics.RNNLLp(tau)))/tau
        exponC = np.exp( -(analytics.R_SL(tau)) -  analytics.R_L(tau))
        results.append(partC*exponC)
    return np.array(results)

    if is_scalar:
        return results[0]  # Return the single result directly
    else:
        return np.array(results)  # Return an array of results


# Numerical integration
CLL0, _ = quad(lambda t: wLL(t), min_tau, max_tau)
CLL1, _ = quad(lambda t: wLL(t) * np.log(t), min_tau, max_tau)
CLL2, _ = quad(lambda t: wLL(t) * np.log(t)**2, min_tau, max_tau)

CNLL0, _ = quad(lambda t: wNLL(t), min_tau, max_tau)
CNLL1, _ = quad(lambda t: wNLL(t) * ((analytics.R_SL(t))), min_tau, max_tau)
CNLL2, _ = quad(lambda t: wNLL(t) * (analytics.R_L(t)), min_tau, max_tau)


#lambda_0 = 0.281920760869
lambda_0 = 0
#lambda_1 = 0.31372693181037903
#lambda_2 = 0.25469261407852173

#lambda_0 = 0
lambda_1 = 0.0
lambda_2 = 0.0

#lambda_1 = 0.0
#lambda_2 = 0.0

def weight(tau_i):
    #return  np.exp(lambda_0 - lambda_1 * np.log(t) - lambda_2 * np.log(t)**2)
    #print("tau=",tau_i)
    if tau_i<10**-10:
       return 0
    Rpt = analytics.Rp(tau_i)
    logFt = analytics.logF(tau_i)
    FpFt = analytics.FpF(tau_i)
    RNNLLpt = analytics.RNNLLp(tau_i)
    part = (CNLL2*Rpt + CNLL1*(RNNLLpt))/((CNLL2-lambda_2)*Rpt+(CNLL1-lambda_1)*(RNNLLpt))
    expon = np.exp( - lambda_1*(analytics.R_SL(tau_i)) -  lambda_2*analytics.R_L(tau_i))
    return expon*part

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
tau_values = np.logspace(-3.0, -0.001, 10000)  # Upper limit set to 0.5 for better visualization

# Calculate theoretical probability densities
pLL_values = wLL(tau_values)
pNLL_values = wNLL(tau_values)
print("pLL_values=", pLL_values)

# Export theoretical distributions to CSV
export_theoretical_data(tau_values, pLL_values, pNLL_values, 'theoretical_distribution.csv')

# Calculating weights for each thrust value
weights = [1 for tau in thrust_values]

# After calculating thrust data
# Export thrust data to CSV
export_thrust_data(thrust_values, weights, 'simulated_thrust.csv')

# After calculating interpolated histogram data
# Bin thrust values for interpolation
linbins = np.linspace(-5.0, -0.001, num=100)
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
plt.hist(np.log10(np.maximum(thrust_values, zeros)), bins=linbins, weights=weights, alpha=0.5, density=True, label='Weighted Simulated Distribution', color='black')

# Unweighted histogram
plt.hist(np.log10(thrust_values), bins=linbins, alpha=0.5, density=True, label='Unweighted Simulated Distribution', color='red')

# Plot interpolated curve
ax.plot(bin_centers, interp_values, label='Interpolated Values (Cubic Spline)', color='orange', linestyle='--')

plt.title('Thrust Distribution with Theoretical Overlay and Interpolation')
plt.xlabel('Log10(Thrust), $\\log_{10}(\\tau)$')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
