import config
from mymath import *
from vector import Vec4

class Particle:

    def __init__(self,pdgid,momentum,col=[0,0],id=-1,cps=[0,0]):
        self.Set(pdgid,momentum,col)
        self.id = id
        self.cps = cps
        self.wgt = []

    def __repr__(self):
        return "{0} {1} {2} {3} {4}".format \
            (self.pid,self.mom,self.col,self.id,self.cps)

    def __str__(self):
        return "{0} {1} {2} {3} {4}".format \
            (self.pid,self.mom,self.col,self.id,self.cps)

    def Set(self,pdgid,momentum,col=[0,0]):
        self.pid = pdgid
        self.mom = momentum
        self.col = col

    def ColorConnected(self,p,w=0):
        if w > 0:
            return (self.col[0] > 0 and self.col[0] == p.col[1])
        if w < 0:
            return (self.col[1] > 0 and self.col[1] == p.col[0])
        return (self.col[0] > 0 and self.col[0] == p.col[1]) or \
               (self.col[1] > 0 and self.col[1] == p.col[0])

    def AddWeight(self,id,t,w):
        self.wgt += [[]]*(id+1-len(self.wgt))
        self.wgt[id].append([t,w])

    def GetWeight(self,w,t):
        if len(self.wgt) == 0: return
        for i in range(len(self.wgt[0])):
            if self.wgt[0][i][0] < t: break
            for j in range(len(w)):
                w[j] *= self.wgt[j][i][1]

def CheckEvent(event,col=1):
    psum = Vec4()
    csum = {}
    for p in event:
        psum += p.mom
        if p.col[0] > 0: 
            csum[p.col[0]] = csum.get(p.col[0],0) + 1
            if csum[p.col[0]] == 0: del csum[p.col[0]]
        if p.col[1] > 0:
            csum[p.col[1]] = csum.get(p.col[1],0) - 1
            if csum[p.col[1]] == 0: del csum[p.col[1]]
    if (abs(psum.E)<config.kin_epsilon and \
        abs(psum.px)<config.kin_epsilon and \
        abs(psum.py)<config.kin_epsilon and \
        abs(psum.pz)<config.kin_epsilon and \
        (len(csum) == 0 if col else True)): return []
    return [ psum, csum ]

