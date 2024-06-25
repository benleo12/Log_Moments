from mymath import *

from vector import Vec4, Rotation, LT
from particle import Particle, CheckEvent
from qcd import AlphaS, NC, TR, CA, CF

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
        A = 2*sik*(pi*n)/D
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
        return self.Ca*(1-z[1])
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

    def UpdateWeights(self,split,evt):
        gsum, osum = mn(0), self.gs[split.id][1]
        self.gs[split.id] = [self.gs[split.id][0],mn(0)]
        for i,sf in enumerate(self.kernels[split.pid]):
            id = self.IDParams(split.mom,evt)
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
        self.UpdateWeights(s[0],event)
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
            self.UpdateWeights(split,event)
        while self.t > self.ct0:
            if em >= nem: return
            self.GeneratePoint(event)
            if em == 0 and self.rt0 != 0:
                self.ct0 = max(self.t0,self.t*self.rt0)
            em += 1
