import torch as m
import math
import random as r
import scipy.special as sp
m.pi = math.pi

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
        self.Bl = 0 # -3./4.
        self.nf = 5

    def T(self,as0,b0,L):
        l = as0*b0*L
        return -1./(m.pi*b0)*m.log(1.-2.*l)

    def Tp(self,as0,b0,L):
        l = as0*b0*L
        return 2.*as0/m.pi/(1.-2.*l)

    def r1(self,as0,b0,a,b,L):
        l = as0*b0*L
        return 1./(2.*m.pi*b0*l*b)*\
            ((a-2.*l)*m.log(1.-2.*l/a)
             -(a+b-2.*l)*m.log(1.-2.*l/(a+b)))

    def lr1p(self,as0,b0,a,b,L):
        return 1./b*(self.T(as0,b0,L/a)-self.T(as0,b0,L/(a+b)))

    def R_LL(self,v):
        L = m.log(1./v)
        return 2.*CF*L*(self.r1(self.as0,self.b0,self.a,self.b,L))

    def R_LLp(self,v):
        L = m.log(1./v)
        return 2.*CF*self.lr1p(self.as0,self.b0,self.a,self.b,L)

    def r2(self,as0,b0,b1,K,a,b,L):
        l = as0*b0*L
        return 1./b*\
            (K/pow(2.*m.pi*b0,2)*\
             ((a+b)*m.log(1.-2.*l/(a+b))-a*m.log(1.-2.*l/a))\
             +b1/(2.*m.pi*pow(b0,3))*\
             (a/2.*pow(m.log(1.-2.*l/a),2)-(a+b)/2*pow(m.log(1.-2.*l/(a+b)),2)\
              +a*m.log(1.-2.*l/a)-(a+b)*m.log(1.-2.*l/(a+b))))

    def r2p(self,as0,b0,b1,K,a,b,L):
        l = as0*b0*L
        return 2.*as0/m.pi*\
            (self.K*l/(2*m.pi*b0)/(a-2*l)/(a+b-2*l)\
             -self.b1/(2*b*b0**2*(a-2*l)*(a+b-2*l))*\
             (a*(a+b-2*l)*m.log(1.-2*l/a)-(a+b)*(a-2*l)*m.log(1-2*l/(a+b))+2*b*l))

    def R_NLL(self,v):
        L = m.log(1./v)
        return 2.*CF*\
            (self.r2(self.as0,self.b0,self.b1,self.K,self.a,self.b,L)+\
             self.Bl*self.T(self.as0,self.b0,L/(self.a+self.b)))

    def R_NLLp(self,v):
        L = m.log(1./v)
        return 2.*CF*\
            (self.r2p(self.as0,self.b0,self.b1,self.K,self.a,self.b,L)+
             self.Bl*self.Tp(self.as0,self.b0,L/(self.a+self.b))/(self.a+self.b))

    def R_NLLc(self,v):
        L = m.log(1./v)
        return 2.*CF*self.r2(self.as0,self.b0,self.b1,self.K,self.a,self.b,L)

    def R_NLLcp(self,v):
        L = m.log(1./v)
        return 2.*CF*self.r2p(self.as0,self.b0,self.b1,self.K,self.a,self.b,L)

    def r(self,as0,b0,b1,K,a,b,L):
        return L*self.r1(as0,b0,a,b,L)+self.r2(as0,b0,b1,K,a,b,L)

    def R(self,v):
        L = m.log(1./v)
        return 2.*CF*\
            (self.r(self.as0,self.b0,self.b1,self.K,self.a,self.b,L)+\
             self.Bl*self.T(self.as0,self.b0,L/(self.a+self.b)))

    def lr1pp(self,as0,b0,a,b,L):
        l = as0*b0*L
        return 2*as0/m.pi/((a-2*l)*(a+b-2*l))

    def R_Lpp(self,v):
        L = m.log(1./v)
        return 2.*CF*self.lr1pp(self.as0,self.b0,self.a,self.b,L)

    def logF(self,v):
        rp = self.R_Lp(v)
        ge = 0.5772156649015329
        return -ge*rp - m.lgamma(1.+rp)

    def Fp(self,v):
        rp = self.R_Lp(v)
        rpp = self.R_Lpp(v)
        ge = 0.5772156649015329
        F = m.exp(self.logF(v))
        return -F*(ge + sp.digamma(1 + rp))*rpp

    def FpF(self,v):
        rp = self.R_Lp(v)
        rpp = self.R_Lpp(v)
        ge = 0.5772156649015329
        return -(ge + sp.digamma(1 + rp))*rpp
