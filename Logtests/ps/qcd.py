from mymath import *
#from mpi4py import MPI

NC = mn(3)
TR = mn(1)/2
CA = NC
CF = (NC*NC-1)/(2*NC)

class AlphaS:

    def __init__(self,mz,asmz,order=1,mb=mn('4.75'),mc=mn('1.3')):
        self.order = order
        self.mc2 = mc*mc
        self.mb2 = mb*mb
        self.mz2 = mz*mz
        self.asmz = asmz
        self.asmb = self(self.mb2)
        self.asmc = self(self.mc2)
#        if MPI.COMM_WORLD.Get_rank() == 0:
        print("\\alpha_s({0}) = {1}".format(mz,self(self.mz2)))

    def beta0(self,nf):
        return mn(11)/6*CA-mn(2)/3*TR*nf

    def beta1(self,nf):
        return mn(17)/6*CA*CA-(mn(5)/3*CA+CF)*TR*nf

    def l2a(self,nf):
        return self.mz2*myexp(-(2*m.pi)/(self.beta0(nf)*self.asmz))

    def asa(self,t,nf):
        tref = self.mz2
        asref = self.asmz
        b0 = self.beta0(nf)/(2*m.pi)
        try:
            return 1/(1/asref+b0*mylog(t/tref))
        except ZeroDivisionError:
            return mn(0)

    def as0(self,t,nf):
        tref = self.mz2
        asref = self.asmz
        b0 = self.beta0(5)/(2*m.pi)
        return 1/(1/asref+b0*mylog(t/tref))

    def as1(self,t,nf):
        tref = self.mz2
        asref = self.asmz
        b0 = self.beta0(nf)/(2*m.pi)
        b1 = self.beta1(nf)/pow(2*m.pi,2)
        w = 1+b0*asref*mylog(t/tref)
        return asref/w*(1-b1/b0*asref*mylog(w)/w)

    def __call__(self,t,nf=-1):
        if self.order == -1: return self.asmz
        if self.order == 0: return self.as0(t,nf)
        return self.as1(t,nf)

