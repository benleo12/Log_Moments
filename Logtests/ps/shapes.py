import math as m

from particle import Particle
from histogram import Histo1D

class ShapeAnalysis:

    def __init__(self,n=1):
        self.n = 0.
        self.sum = 0.
        self.thisto = [ Histo1D(120,-2.,0.,'/EventShapes/v\n') for i in range(n+1) ]

    def FillHistos(self,s,w,weights=[]):
        self.n += 1.
        self.sum += w
        tau = m.log10(s) if s>0. else -1.e37
        self.thisto[0].Fill(tau,w)
        for i in range(len(weights)):
            self.thisto[i+1].Fill(tau,w*weights[i])

    def FillIntHistos(self,s,w,weights=[]):
        self.n += 1.
        self.sum += w
        tau = m.log10(1.-s) if s<1. else -1.e37
        self.thisto[0].FillFrom(tau,w)
        for i in range(len(weights)):
            self.thisto[i+1].FillFrom(tau,w*weights[i])

    def Finalize(self,name,scale=1.):
        self.thisto[0].ScaleW(scale/self.sum)
        file = open(name+".yoda","w")
        file.write(str(self.thisto[0]))
        file.close()
        for i in range(1,len(self.thisto)):
            self.thisto[i].ScaleW(scale/self.sum)
            file = open(name+".{0}.yoda".format(i),"w")
            file.write(str(self.thisto[i]))
            file.close() 
