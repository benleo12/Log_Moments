
import math as m
import random as r

from vector import Vec4
from particle import Particle

class eetojj:

    def __init__(self,alphas,ecms=91.2):
        self.alphas = alphas
        self.ecms = ecms
        self.MZ2 = pow(91.1876,2.)
        self.GZ2 = pow(2.4952,2.)
        self.alpha = 1./128.802
        self.sin2tw = 0.22293

    def ME2(self,fl,s,t):
        qe = -1.
        ae = -0.5
        ve = ae - 2.*qe*self.sin2tw;
        qf = 2./3. if fl in [2,4] else -1./3.
        af = 0.5 if fl in [2,4] else -0.5
        vf = af - 2.*qf*self.sin2tw;
        kappa = 1./(4.*self.sin2tw*(1.-self.sin2tw))
        chi1 = kappa * s * (s-self.MZ2)/(pow(s-self.MZ2,2.) + self.GZ2*self.MZ2);
        chi2 = pow(kappa * s,2.)/(pow(s-self.MZ2,2.) + self.GZ2*self.MZ2);
        term1 = (1+pow(1.+2.*t/s,2.))*(pow(qf*qe,2.)+2.*(qf*qe*vf*ve)*chi1+(ae*ae+ve*ve)*(af*af+vf*vf)*chi2)
        term2 = (1.+2.*t/s)*(4.*qe*qf*ae*af*chi1+8.*ae*ve*af*vf*chi2)
        return pow(4.*m.pi*self.alpha,2.)*3.*(term1+term2)

    def GeneratePoint(self):
        ct = 2.*r.random()-1.
        st = m.sqrt(1.-ct*ct)
        phi = 2.*m.pi*r.random()
        p1 = Vec4(1,st*m.cos(phi),st*m.sin(phi),ct)*self.ecms/2 
        p2 = Vec4(p1.E,-p1.px,-p1.py,-p1.pz)
        pa = Vec4(self.ecms/2,0,0,self.ecms/2)
        pb = Vec4(self.ecms/2,0,0,-self.ecms/2)
        fl = r.randint(1,5)
        lome = self.ME2(fl,(pa+pb).M2(),(pa-p1).M2())
        dxs = 5.*lome*3.89379656e8/(8.*m.pi)/(2.*pow(self.ecms,2))
        return ( [
            Particle(-11,-pa),
            Particle(11,-pb),
            Particle(fl,p1,[1,0]),
            Particle(-fl,p2,[0,1])
        ], dxs, lome )

    def GenerateLOPoint(self):
        lo = self.GeneratePoint()
        return ( lo[0], lo[1] )
