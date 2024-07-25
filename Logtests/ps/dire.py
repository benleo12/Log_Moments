from mymath import *

from vector import Vec4, Rotation, LT
from particle import Particle, CheckEvent
from qcd import AlphaS, NC, TR, CA, CF

K = (67./18.-pow(m.pi,2)/6.)*CA-10./9.*TR*5

class Kernel:

    def __init__(self,flavs,Ca,alpha,t0):
        self.flavs = flavs
        self.Ca = Ca
        self.alphamax=alpha(t0)
        self.alpha=alpha

class Soft (Kernel):

    def Value(self,z,k2,t):
        return self.Ca*(2*(-z[1])/(z[1]*z[1]+k2)-2)*(1+self.alpha(t)/(2*m.pi)*K)/(z[0]+z[1])
    def Estimate(self,z,k02):
        return self.Ca*2*(-z[1])/(z[1]*z[1]+k02)*(1+self.alphamax/(2*m.pi)*K)
    def Integral(self,k02):
        return self.Ca*mylog(1+1/k02)*(1+self.alphamax/(2*m.pi)*K)
    def GenerateZ(self,k02):
        return [mn(1),-mysqrt(k02*(mypow(1+1/k02,mn(rng.random()))-1))]

class Cqq (Kernel):

    def Value(self,z,k2,t):
        return self.Ca*(1-z[1])*0
    def Estimate(self,z,k02):
        return self.Ca
    def Integral(self,k02):
        return self.Ca
    def GenerateZ(self,k02):
        return [mn(0),mn(rng.random())]

class Cgg (Kernel):

    def Value(self,z,k2,t):
        return self.Ca*z[1]*(1-z[1])*0
    def Estimate(self,z,k02):
        return self.Ca
    def Integral(self,k02):
        return self.Ca
    def GenerateZ(self,k02):
        return [mn(0),mn(rng.random())]

class Cgq (Kernel):

    def Value(self,z,k2,t):
        return TR/2*(1-2*z[1]*(1-z[1]))*0
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
            self.kernels[fl] = [ Soft([fl,fl,21],CA/2 if lc else CF,alpha[1],self.t0) ]
        self.kernels[21] = [ Soft([21,21,21],CA/2,alpha[1],self.t0) ]
        if coll & 1:
            for fl in [-5,-4,-3,-2,-1,1,2,3,4,5]:
                self.kernels[fl].append( Cqq([fl,fl,21],CA/2 if lc else CF,alpha[1],self.t0) )
        if coll & 2:
            self.kernels[21].append( Cgg([21,21,21],CA/2,alpha[1],self.t0) )
            self.kernels[21].append( Cgq([21,0,0],0,alpha[1],self.t0) )
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
        w *= s[2].Value(z,kt**2/s[3],kt**2)/s[2].Estimate(z,self.ct0/s[3])
        w *= 1/(1+self.beta)
        if w <= rng.random(): return False
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
            if em == 0 and self.rt0 != 0:
                self.ct0 = max(self.t0,self.t*self.rt0)
            em += 1


if __name__== "__main__":

    import sys, time, optparse

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
    parser.add_option("-x","--piece",default="all",dest="piece")
    (opts,args) = parser.parse_args()

    opts.histo = opts.histo.format(**vars(opts))
    if opts.logfile != "":
        sys.stdout = open(opts.logfile, 'w')

    import config
    config.quad_precision = int(opts.quad)
    from mymath import *
    print_math_settings()

    from vector import Vec4, Rotation, LT
    from particle import Particle, CheckEvent
    from qcd import AlphaS, NC, TR, CA, CF
    from analysis import SimplifiedAnalysis

    ecms = mn(opts.ecms)
    lam = mn(opts.asmz)/mn(opts.alphas)
    t0 = mypow(mn(opts.cut)/ecms**2,lam)*ecms**2

    if opts.piece == 'll':
        K=0
        opts.coll=0
        opts.order=0
        alphas = [ AlphaS(ecms,mn(opts.alphas),0,mb=1e-3,mc=1e-4),
                   AlphaS(ecms,mn(opts.alphas),0,mb=1e-3,mc=1e-4) ]
    if opts.piece == 'nllc':
        opts.coll=0
        opts.order=1
        alphas = [ AlphaS(ecms,mn(opts.alphas),1,mb=1e-3,mc=1e-4),
                   AlphaS(ecms,mn(opts.alphas),0,mb=1e-3,mc=1e-4) ]
    else:
        alphas = [ AlphaS(ecms,mn(opts.alphas),int(opts.order),mb=1e-3,mc=1e-4),
                   AlphaS(ecms,mn(opts.alphas),int(opts.order),mb=1e-3,mc=1e-4) ]

    print("t_0 = {0}, log(Q^2/t_0) = {1}, \\alpha_s(t_0) = {2} / {3}". \
          format(t0,mylog(ecms**2/t0),alphas[0](t0),alphas[1](t0)))
    shower = Shower(alphas,t0,int(opts.coll),mn(opts.beta),
                    mn(opts.rcut),eval(opts.flat),int(opts.nmax),opts.lc)
    jetrat = SimplifiedAnalysis(-0.0033)

    rng.seed(int(opts.seed))
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

    import csv
    with open('thrust_values.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for value in thrust_values:
            writer.writerow([value])
