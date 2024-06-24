import math as m
import random as r

from vector import Vec4
from particle import Particle
from qcd import AlphaS, NC, TR, CA, CF

class Pqq:

    def __init__(self,alpha):
        self.alpha = alpha

    def Value(self,z,t,Q2,v,a,b):
        kt2 = t*pow(1.-z,2.*b/(a+b))
        if pow(1.-z,2) < kt2/Q2: return 0.
        as_soft = self.alpha(kt2)/(2.*m.pi)
        return as_soft*CF*2./(1.-z)

    def Estimate(self,z,eps):
        if z > 1.-eps: return 0.
        as_max = self.alpha(1.)/(2.*m.pi)
        return as_max*CF*2./(1.-z)

    def Integral(self,eps):
        as_max = self.alpha(1.)/(2.*m.pi)
        return as_max*CF*2.*m.log(1./eps)

    def GenerateZ(self,eps):
        ran = r.random()
        return 1.-m.pow(eps,ran)

class Shower:

    def __init__(self,alpha,vc,a,b):
        self.a = a
        self.b = b
        self.vc = vc
        self.kernel = Pqq(alpha)

    def TC(self):
        return self.q2*pow(self.vc*self.v,2./(self.a+self.b))

    def GeneratePoint(self,event):
        while self.t > self.TC():
            t = self.TC()
            for split in event[2:4]:
                for spect in event[2:4]:
                    if spect == split: continue
                    eps = pow(self.TC()/self.q2,(self.a+self.b)/(2.*self.a))
                    G = self.kernel.Integral(eps)
                    tt = self.t*m.pow(r.random(),1./G)
                    if tt > t:
                        t = tt
                        s = [ split, spect, eps ]
            self.t = t
            if t == self.TC(): return 1.
            z = self.kernel.GenerateZ(s[2])
            f = self.kernel.Value(z,t,self.q2,self.v,self.a,self.b)
            g = self.kernel.Estimate(z,s[2])
            if f/g > r.random():
                vi = pow(t/self.q2,(self.a+self.b)/2.)
                if vi > self.v:
                    return 0.
                return 1.
    
    def Run(self,event,t):
        self.t = t
        self.q2 = t
        self.v = pow(10.,-2.*r.random())
        return 2. * self.GeneratePoint(event)
            
# build and run the generator

import sys, optparse, matrix, shapes

parser = optparse.OptionParser()
parser.add_option("-s","--seed",default=123456,dest="seed")
parser.add_option("-e","--events",default=1000000,dest="events")
parser.add_option("-f","--file",default="llps",dest="histo")
parser.add_option("-c","--cutoff",default=0.001,dest="vc")
parser.add_option("-a","--a_coefficient",default=1,dest="a")
parser.add_option("-b","--b_coefficient",default=1,dest="b")
(opts,args) = parser.parse_args()

alphas = AlphaS(91.2,0.118,0)
hardxs = matrix.eetojj()
shower = Shower(alphas,vc=float(opts.vc),a=float(opts.a),b=float(opts.b))
shapes = shapes.ShapeAnalysis()

r.seed(int(opts.seed))
for i in range(int(opts.events)):
    event, weight = hardxs.GenerateLOPoint()
    w = shower.Run(event,pow(91.2,2))
    if i%100==0: sys.stdout.write('\rEvent {0}'.format(i))
    sys.stdout.flush()
    shapes.FillHistos(shower.v,weight,[w])
shapes.Finalize(opts.histo)
print ""
