import math as m
import random as r

from vector import Vec4
from qcd import AlphaS, NC, TR, CA, CF

class LL:

    def __init__(self,alpha,a,b):
        self.alpha = alpha
        self.a = a
        self.b = b

    def r1(self,as0,b0,a,b,L):
        l = as0*b0*L
        return 1./(2.*m.pi*b0*l*b)*\
            ((a-2.*l)*m.log(1.-2.*l/a)
             -(a+b-2.*l)*m.log(1.-2.*l/(a+b)))
        
    def r(self,as0,b0,a,b,L):
        return L*self.r1(as0,b0,a,b,L)

    def R(self,v):
        L = m.log(1./v)
        return 2.*CF*self.r(self.as0,self.b0,self.a,self.b,L)

    def Run(self,event,t):
        self.as0 = self.alpha(t)
        self.b0 = self.alpha.beta0(5)/(2.*m.pi)
        self.v = pow(10.,-2.*r.random())
        return 2. * m.exp(-self.R(self.v))
            
# build and run the generator

import sys, optparse, matrix, shapes

parser = optparse.OptionParser()
parser.add_option("-e","--events",default=1000000,dest="events")
parser.add_option("-f","--file",default="ll",dest="histo")
parser.add_option("-a","--a_coefficient",default=1,dest="a")
parser.add_option("-b","--b_coefficient",default=1,dest="b")
(opts,args) = parser.parse_args()

alphas = AlphaS(91.2,0.118,0)
hardxs = matrix.eetojj()
shower = LL(alphas,a=float(opts.a),b=float(opts.b))
shapes = shapes.ShapeAnalysis()

for i in range(int(opts.events)):
    event, weight = hardxs.GenerateLOPoint()
    w = shower.Run(event,pow(91.2,2))
    if i%100==0: sys.stdout.write('\rEvent {0}'.format(i))
    sys.stdout.flush()
    shapes.FillHistos(shower.v,weight,[w])
shapes.Finalize(opts.histo)
print ""
