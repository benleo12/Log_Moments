import torch as m
import random as r
import scipy.special as sp

from vector import Vec4
from qcd import AlphaS, NC, TR, CA, CF

class NLL:

    def __init__(self,alpha,a,b,t):
        self.alpha = alpha
        self.a = a
        self.b = b
        self.as0 = self.alpha[0](t)
        self.b0 = self.alpha[0].beta0(5)/(2.*m.pi)
        self.b1 = self.alpha[1].beta1(5)/pow(2.*m.pi,2)
        self.K = (67./18.-pow(m.pi,2)/6.)*CA-10./9.*TR*5
        self.nf = 5

    def r1(self,as0,b0,a,b,L):
        l = as0*b0*L
        return 1./(2.*m.pi*b0*l*b)*\
            ((a-2.*l)*m.log(1.-2.*l/a)
             -(a+b-2.*l)*m.log(1.-2.*l/(a+b)))

    def r2(self,as0,b0,b1,K,a,b,L):
        l = as0*b0*L
        return 1./b*\
            (K/pow(2.*m.pi*b0,2)*\
             ((a+b)*m.log(1.-2.*l/(a+b))-a*m.log(1.-2.*l/a))\
             +b1/(2.*m.pi*pow(b0,3))*\
             (a/2.*pow(m.log(1.-2.*l/a),2)-(a+b)/2*pow(m.log(1.-2.*l/(a+b)),2)\
              +a*m.log(1.-2.*l/a)-(a+b)*m.log(1.-2.*l/(a+b))))

    def r(self,as0,b0,b1,K,a,b,L):
        return L*self.r1(as0,b0,a,b,L)+self.r2(as0,b0,b1,K,a,b,L)

    def T(self,as0,b0,L):
        l = as0*b0*L
        #print("l and b0", l, b0, L)
        return -1./(m.pi*b0)*m.log(1.-2.*l)

    def R(self,v):
        L = m.log(1./v)
        Bl = -3./4.
        return 2.*CF*\
            (self.r(self.as0,self.b0,self.b1,self.K,self.a,self.b,L)+\
             Bl*self.T(self.as0,self.b0,L/(self.a+self.b)))

    def R_L(self,v):
        L = m.log(1./v)
        A1 = 2.*CF
        return A1*\
            L*(self.r1(self.as0,self.b0,self.a,self.b,L))

    def R_SL(self,v):
        L = m.log(1./v)
        Bl = -3./4.
        A1 = 2.*CF
        return A1*\
            (self.r2(self.as0,self.b0,self.b1,self.K,self.a,self.b,L)+\
             Bl*self.T(self.as0,self.b0,L/(self.a+self.b)))

    def R_NLLc(self,v):
        L = m.log(1./v)
        A1 = 2.*CF
        return A1*\
            (self.r2(self.as0,self.b0,self.b1,self.K,self.a,self.b,L))

    def rp(self,as0,b0,a,b,L):
        return 1./b*(self.T(as0,b0,L/a)-self.T(as0,b0,L/(a+b)))

    def rpp(self,as0,b0,a,b,L):
        l = as0*b0*L
        return (as0*2/m.pi)*1./((a-2*l)*(a+b-2*l))

    def rnnllp(self,as0,b0,b1,a,b,L,nf):
        l = as0*b0*L
        A1 = 2.*CF*(2*as0/m.pi)
        A2 = CF*(CA*(67/9-m.pi**2/3)-10/9*nf)*(2*as0/m.pi)
        Bl = -3./4.
        Bp = Bl*1/(a+b-2*l)
        return (-A1*b1*(a*(a+b-2*l)*m.log(1.-2*l/a)-(a+b)*(a-2*l)*m.log(1-2*l/(a+b))+2*b*l) /( 2*b*b0**2*(a-2*l)*(a+b-2*l) ) + A2*l/(2*m.pi*b0)/(a-2*l)/(a+b-2*l)) + A1*Bp

    def rnllcp(self,as0,b0,b1,a,b,L,nf):
        l = as0*b0*L
        A1 = 2.*CF*(2*as0/m.pi)
        A2 = CF*(CA*(67/9-m.pi**2/3)-10/9*nf)*(2*as0/m.pi)
        Bl = 0
        Bp = Bl*1/(a+b-2*l)
        return (-A1*b1*(a*(a+b-2*l)*m.log(1.-2*l/a)-(a+b)*(a-2*l)*m.log(1-2*l/(a+b))+2*b*l) /( 2*b*b0**2*(a-2*l)*(a+b-2*l) ) + A2*l/(2*m.pi*b0)/(a-2*l)/(a+b-2*l)) + A1*Bp


    def Rp(self,v):
        L = m.log(1./v)
        return 2.*CF*self.rp(self.as0,self.b0,self.a,self.b,L)

    def RNNLLp(self,v):
        L = m.log(1./v)
        return self.rnnllp(self.as0,self.b0,self.b1,self.a,self.b,L,self.nf)

    def RNLLcp(self,v):
        L = m.log(1./v)
        return self.rnllcp(self.as0,self.b0,self.b1,self.a,self.b,L,self.nf)

    def Rpp(self,v):
        L = m.log(1./v)
        A1 = 2*CF
        return 2.*CF*self.rpp(self.as0,self.b0,self.a,self.b,L)

    def logF(self,v):
        rp = self.Rp(v)
        ge = 0.5772156649015329
        return -ge*rp - m.lgamma(1.+rp)

    def Fp(self,v):
        rp = self.Rp(v)
        rpp = self.Rpp(v)
        ge = 0.5772156649015329
        F = m.exp(self.logF(v))
        return -F*(ge + sp.digamma(1 + rp))*rpp

    def FpF(self,v):
        rp = self.Rp(v)
        rpp = self.Rpp(v)
        ge = 0.5772156649015329
        #F = m.exp(self.logF(v))
        return -(ge + sp.digamma(1 + rp))*rpp

    def Run(self,event,t):
        self.as0 = self.alpha[0](t)
        self.b0 = self.alpha[0].beta0(5)/(2.*m.pi)
        self.b1 = self.alpha[1].beta1(5)/pow(2.*m.pi,2)
        self.K = (67./18.-pow(m.pi,2)/6.)*CA-10./9.*TR*5
        #self.v = pow(10.,-2.*r.random())
        #return 2. * m.exp(-self.R(self.v))*self.F(self.v)

# build and run the generator

#import sys, optparse, matrix, shapes

#parser = optparse.OptionParser()
#parser.add_option("-e","--events",default=1000000,dest="events")
#parser.add_option("-f","--file",default="nll",dest="histo")
#parser.add_option("-a","--a_coefficient",default=1,dest="a")
#parser.add_option("-b","--b_coefficient",default=1,dest="b")
#(opts,args) = parser.parse_args()

#alphas = [ AlphaS(91.2,0.118,0), AlphaS(91.2,0.118,1) ]
#hardxs = matrix.eetojj()
#shower = NLL(alphas,a=float(opts.a),b=float(opts.b))
#shapes = shapes.ShapeAnalysis()

#for i in range(int(opts.events)):
#    event, weight = hardxs.GenerateLOPoint()
#    w = shower.Run(event,pow(91.2,2))
#    if i%100==0: sys.stdout.write('\rEvent {0}'.format(i))
#    sys.stdout.flush()
#    shapes.FillHistos(shower.v,weight,[w])
#shapes.Finalize(opts.histo)
