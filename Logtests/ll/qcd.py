
import math as m

NC = 3.
TR = 1./2.
CA = NC
CF = (NC*NC-1.)/(2.*NC)

class AlphaS:

    def __init__(self,mz,asmz,order=1):
        self.order = order
        self.mz2 = mz*mz
        self.asmz = asmz
        print "\\alpha_s({0}) = {1} ({2} loop)".format(mz,self(self.mz2),order+1)

    def beta0(self,nf):
        return 11./6.*CA-2./3.*TR*nf

    def beta1(self,nf):
        return 17./6.*CA*CA-(5./3.*CA+CF)*TR*nf

    def as0(self,t):
        tref = self.mz2
        asref = self.asmz
        b0 = self.beta0(5)/(2.*m.pi)
        return 1./(1./asref+b0*m.log(t/tref))

    def as1(self,t):
        tref = self.mz2
        asref = self.asmz
        b0 = self.beta0(5)/(2.*m.pi)
        b1 = self.beta1(5)/pow(2.*m.pi,2)
        w = 1.+b0*asref*m.log(t/tref)
        if w < 0.:
            print '\\alpha_s out of bounds at t = ',t
            return 0.
        return asref/w*(1.-b1/b0*asref*m.log(w)/w)

    def __call__(self,t):
        if self.order == 0: return self.as0(t)
        return self.as1(t)
