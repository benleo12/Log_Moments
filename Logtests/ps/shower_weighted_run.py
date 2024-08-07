import math as m
import random as r
import nll
from vector import Vec4
from particle import Particle, CheckEvent
from qcd import AlphaS, NC, TR, CA, CF
import numpy as np



class Soft:

    def __init__(self,flavs,Ca):
        self.flavs = flavs
        self.Ca = Ca

    def Value(self,z,pi,pj,pk,e):
        n = -pj-e[0]-e[1]
        sij = pi.SmallMLDP(pj)
        sik = pi.SmallMLDP(pk)
        skj = pk.SmallMLDP(pj)
        D = sij*(pk*n)+skj*(pi*n)
        if D == 0: return mn(0)
        A = 2*sik*(pi*n)/D/2
        return self.Ca*A
    def Estimate(self,z,ip):
        return self.Ca*4/(-z[1])
    def Integral(self,ip):
        return self.Ca*4*mylog(1/ip[-1])
    def GenerateZ(self,ip):
        return [mn(1),-mypow(ip[-1],rng.random())]

class Coll:

    def __init__(self,flavs,Ca):
        self.flavs = flavs
        self.Ca = Ca

class Cqq (Coll):

    def Value(self,z,pi,pj,pk,e):
        return self.Ca*(1-z[1])*0
    def Estimate(self,z,ip):
        return self.Ca
    def Integral(self,ip):
        return self.Ca
    def GenerateZ(self,ip):
        return [mn(0),mn(rng.random())]

class Cgg (Coll):

    def Value(self,z,pi,pj,pk,e):
        return self.Ca*z[1]*(1-z[1])
    def Estimate(self,z,ip):
        return self.Ca
    def Integral(self,ip):
        return self.Ca
    def GenerateZ(self,ip):
        return [mn(0),mn(rng.random())]

class Cgq (Coll):

    def Value(self,z,pi,pj,pk,e):
        return TR/2*(1-2*z[1]*(1-z[1]))
    def Estimate(self,z,ip):
        return TR/2
    def Integral(self,ip):
        return 5*TR/2
    def GenerateZ(self,ip):
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
        self.alpha = alpha
        self.alphamax = alpha(self.t0)
        self.amode = 0 if self.alpha.order == -1 else 1
        if self.amode != 0:
            self.alphamax = (2*m.pi)/self.alpha.beta0(5)
        self.kernels = {}
        for fl in [-5,-4,-3,-2,-1,1,2,3,4,5]:
            self.kernels[fl] = [ Soft([fl,fl,21],CA/2 if lc else CF)  ]
        self.kernels[21] = [ Soft([21,21,21],CA/2) ]
        if coll & 1:
            for fl in [-5,-4,-3,-2,-1,1,2,3,4,5]:
                self.kernels[fl].append( Cqq([fl,fl,21],CA/2 if lc else CF) )
        if coll & 2:
            self.kernels[21].append( Cgg([21,21,21],CA/2) )
            self.kernels[21].append( Cgq([21,0,0],0) )
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
                     (cx**2/mn(2)-2*self.Bl*(cx+13/mn(4)-30*v)))* \
            (-J*aS*self.Ca/(m.pi)*(cx-2*self.Bl*(1-30*v)))

    def MakeKinematics(self,x,y,phi,pijt,pkt,moms):
        Kt = moms[0]+moms[1]
        v, Q2 = -y, 2*pijt*Kt
        z, omz = x[0]+x[1], 1-x[1] if x[0] == 0 else -x[1]
        pk, pi = pkt, pijt*z
        n, n_perp = Kt+pijt-pi, Rotation(pi,pk).y
        if n_perp.P2() == 0: return []
        n_perp *= 1/n_perp.P()
        l_perp = LT(n,pi,n_perp)
        if l_perp.M2() == 0: l_perp = pi.Cross(n_perp)
        l_perp *= 1/mysqrt(abs(l_perp.M2()))
        kap = Kt.M2()/Q2
        ktt = v*((1-v)*omz-v*kap)*Q2
        if ktt < 0: return []
        ktt = mysqrt(ktt);
        pj = ktt*(mycos(phi)*n_perp+mysin(phi)*l_perp)
        pj += omz*pijt+v*(Kt-(omz+2*kap)*pijt)
        K = Kt+pijt-pi-pj
        oldcm, newcm = -Kt, -K
        if oldcm.M2() <= 0 or newcm.M2() <= 0: return []
        for i,p in enumerate(moms):
            if i<2: continue
            moms[i] = oldcm.BoostBack(newcm.Boost(p))
        pi = oldcm.BoostBack(newcm.Boost(pi))
        pj = oldcm.BoostBack(newcm.Boost(pj))
        pk = oldcm.BoostBack(newcm.Boost(pk))
        return [pi,pj,pk,moms]

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

    def IDParams(self,qa,evt):
        Kt = -evt[0].mom-evt[1].mom
        Q2 = 2*qa*Kt
        kap = Kt.M2()/Q2
        t0 = self.ct0/Kt.M2()
        omzp = t0/(1+t0)
        return [Kt,Q2,kap,omzp]

    def UpdateWeights(self,split):
        gsum, osum = mn(0), self.gs[split.id][1]
        self.gs[split.id] = [self.gs[split.id][0],mn(0)]
        for i,sf in enumerate(self.kernels[split.pid]):
            id = self.IDParams(split.mom,event)
            g = self.alphamax/(2*m.pi)*sf.Integral(id)
            ng = 2 if ( split.col[0]!=0 and split.col[1]!=0 ) else 1
            self.gs[split.id].append([gsum+ng*g,split.id,ng,i,id])
            gsum += ng*g
        self.gs[split.id][1] = gsum
        for g in self.gs[split.id:]: g[0] += gsum-osum

    def GenerateZ(self,event,momsum,s,t):
        z = s[2].GenerateZ(s[3])
        omz = 1-z[1] if z[0] == 0 else -z[1]
        if omz == 0: return False
        phi = 2*m.pi*rng.random()
        Q2, sijt = momsum.M2(), 2*s[0].mom.SmallMLDP(s[1].mom)
        sit, sjt = 2*momsum*s[0].mom, 2*momsum*s[1].mom
        if sijt <= 0 or sit >= 0 or sjt >= 0: return False
        v, rho = mysqrt(t/Q2), mypow(sit*sjt/(Q2*sijt),self.beta/2)
        Q, a, b = mysqrt(sijt/Q2*sit/sjt), 1, self.beta
        kt = mysqrt(Q2)*mypow(rho*v,a/(a+b))*mypow(Q*omz,b/(a+b))
        y = kt**2/(-2*s[0].mom*momsum)
        y = y/omz
        evt = [ p.mom for p in event ]
        moms = self.MakeKinematics(z,y,phi,s[0].mom,s[1].mom,evt)
        if moms == []: return False
        if self.amode == 0:
            w = self.alpha(kt**2,5)/self.alphamax
        else:
            asref = self.alpha.asa(t,5)
            if asref>0: w = self.alpha(kt**2,5)/asref
            else: w = 0
        w *= s[2].Value(z,moms[0],moms[1],moms[2],moms[3])/s[2].Estimate(z,s[3])
        w *= 1/(1+self.beta)
        if w <= rng.random(): return False
        for i, j in zip(event,evt): i.mom = j
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
        self.UpdateWeights(s[0])
        self.gs.append([self.gs[-1][0],mn(0)])
        self.UpdateWeights(s[2])
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
        if k != self.gs[k][r][1]: print('Error 1 in integral table')
        if event[k].cps[0] != 0:
            if event[k].cps[1] != 0:
                if self.gs[k][r][2] != 2: print('Error 2 in integral table')
                s = rng.choice(event[k].cps)
            else:
                if self.gs[k][r][2] != 1: print('Error 2 in integral table')
                s = event[k].cps[0]
        else:
            if self.gs[k][r][2] != 1: print('Error 2 in integral table')
            s = event[k].cps[1]
        sf = self.kernels[event[k].pid][self.gs[k][r][-2]]
        return [ event[k], event[s], sf, self.gs[k][r][-1] ]

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
            self.UpdateWeights(split)
        while self.t > self.ct0:
            if em >= nem: return
            self.GeneratePoint(event)
            if em == 0 and self.rt0 != 0:
                self.ct0 = max(self.t0,self.t*self.rt0)
            em += 1

# build and run the generator
import sys, time, optparse
#from mpi4py import MPI
print("print here")
parser = optparse.OptionParser()
parser.add_option("-s","--seed",default=123456,dest="seed")
parser.add_option("-e","--events",default=1000,dest="events")
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
alphas = AlphaS(ecms,mn(opts.alphas),int(opts.order))
#if MPI.COMM_WORLD.Get_rank() == 0:
print("t_0 = {0}, log(Q^2/t_0) = {1}, \\alpha_s(t_0) = {2}". \
          format(t0,mylog(ecms**2/t0),alphas(t0)))
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


import matplotlib.pyplot as plt
from scipy.integrate import quad
# Integration range
min_tau = 10**-6#10
max_tau = 0.999#0.1
#coeff = 2 * alpha_s / (3 * np.pi)

alphas = [ AlphaS(91.2,0.03,0), AlphaS(91.2,0.03,1) ]
analytics = nll.NLL(alphas,a=1,b=1,t=91.2**2)


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
CLL2, _ = quad(lambda t: wLL(t) * np.log(t)**2, min_tau, max_tau)

CNLL0, _ = quad(lambda t: wNLL(t), min_tau, max_tau)
CNLL1, _ = quad(lambda t: wNLL(t) * ((analytics.R_SL(t)-analytics.logF(t))), min_tau, max_tau)
CNLL2, _ = quad(lambda t: wNLL(t) * (analytics.R_L(t)), min_tau, max_tau)


#lambda_0 = 0.281920760869
lambda_0 = 0
#lambda_1 = 0.31372693181037903
#lambda_2 = 0.25469261407852173

#lambda_0 = 0
lambda_1 = 0.44
lambda_2 = 0.5

#lambda_1 = 0.0
#lambda_2 = 0.0

def weight(tau_i):
    #return  np.exp(lambda_0 - lambda_1 * np.log(t) - lambda_2 * np.log(t)**2)
    #print("tau=",tau_i)
    if tau_i<10**-12:
       return 0
    Rpt = analytics.Rp(tau_i)
    logFt = analytics.logF(tau_i)
    FpFt = analytics.FpF(tau_i)
    RNNLLpt = analytics.RNNLLp(tau_i)
    part = (CNLL2*Rpt + CNLL1*(RNNLLpt-FpFt))/((CNLL2-lambda_2)*Rpt+(CNLL1-lambda_1)*(RNNLLpt-FpFt))
    expon = np.exp(lambda_0 - lambda_1*(analytics.R_SL(tau_i)-analytics.logF(tau_i)) -  lambda_2*analytics.R_L(tau_i))
    return expon*part

# Generate tau values, avoiding 0 to prevent division by zero or log(0)
tau_values = np.logspace(-10, -0.001, 1000)  # Upper limit set to 0.5 for better visualization

# Calculate theoretical probability densities
pLL_values = wLL(tau_values)
pNLL_values = wNLL(tau_values)
print("pLL_values=",pLL_values)
# Now plot the theoretical distribution
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1,1,1)
ax.plot(np.log10(tau_values), np.log(10)*tau_values*pLL_values, label='LL Theoretical Thrust Distribution', color='blue')
ax.plot(np.log10(tau_values), np.log(10)*tau_values*pNLL_values, label='NLL Theoretical Thrust Distribution', color='green')
#ax.set_yscale('log')
#ax.set_xscale('log')
# Overlay with actual thrust distribution
# Assuming 'thrust_values' is a list of thrust values from the previous operations
linbins = np.linspace(-12,0.0,num=100)
#print("logbins=",logbins)

# Calculating weights for each thrust value
weights = [weight(tau) for tau in thrust_values]

# Weighted histogram
print("log10vals=",np.log10(thrust_values))
zeros=np.ones(len(thrust_values))*10**-24
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
