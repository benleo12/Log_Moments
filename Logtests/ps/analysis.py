from mymath import *
from histogram import Histo1D
import durham, cambridge
from hemispheres import Hemispheres
from vector import Vec4
#from mpi4py import MPI

class SimplifiedAnalysis:
    def __init__(self, lam):
        self.lam = lam / 2
        self.thrust_values = []  # List to store thrust values directly
        self.weight_values = []

    def logcut(self, v, cut):
        if v == 0 or v < myexp(cut): return 1
        return mylog(v)

    def Analyze(self, event, w):
        hems = Hemispheres(event[2:])
        thrust = hems.Tau()
        if thrust<=0.0: return
        self.thrust_values.append(thrust)  # Store thrust value directly
        self.weight_values.append(w)

    def Finalize(self):
        return self.thrust_values, self.weight_values

class Analysis:

    def __init__(self,lam,cas,n=0):
        self.n = mn(0)
        self.mr = -0.6
        self.lam = lam/2
        self.cas = cas
        self.weight = mn(0)
        self.ynm = [ Histo1D(120,self.mr,0.0,'/LL_JetRates/log10_y_{0}{1}\n'.format(i+2,i+3)) for i in range(4) ]
        self.duralg = durham.Algorithm()
        self.camalg = cambridge.Algorithm()
        pname = '/LL_EvtShp/{SHAPE}'
        self.EvtShp = {shape: Histo1D(120,self.mr,0.0,pname.format(SHAPE=shape))
                            for shape in ["Thrust","HeavyMassSq","LightMassSq","SingleMassSq",
                                          "HeavyMinusLightMassSq","TotalBroad","WideBroad",
                                          "NarrowBroad","SingleBroad","FC1.5","FC1","FC0.5","FC0","y23"]}
        self.EvtShp["DPsi"] = Histo1D(64,0.0,1.0,pname.format(SHAPE="DPsi"))
        self.mr *= self.lam

    def log10cut(self,v,cut):
        if v == 0 or v < mypow(10,cut): return 1
        return mylog10(v)

    def logcut(self,v,cut):
        if v == 0 or v < myexp(cut): return 1
        return mylog(v)

    def Analyze(self,event,w,weights=[]):
        self.n += 1
        self.weight += w
        if self.cas == -1: return
        if self.cas & 2:
            kt2 = self.duralg.Cluster(event)
            for j in range(len(self.ynm)):
                self.ynm[j].FillNextFrom(self.logcut(kt2[-1-j],-1/self.lam)*self.lam if len(kt2)>j else -5.,w)
        hems = Hemispheres(event[2:])
        self.EvtShp["Thrust"].FillNextFrom(self.logcut(hems.Tau(),-1/self.lam)*self.lam*2,w)
        if self.cas & 4:
            self.EvtShp["HeavyMassSq"].FillNextFrom(self.logcut(hems.HeavyMassSq(),-1/self.lam)*self.lam*2,w)
            self.EvtShp["LightMassSq"].FillNextFrom(self.logcut(hems.LightMassSq(),-1/self.lam)*self.lam*2,w)
            self.EvtShp["HeavyMinusLightMassSq"].FillNextFrom(self.logcut(hems.HeavyMinusLightMassSq(),-1/self.lam)*self.lam*2,w)
        self.EvtShp["TotalBroad"].FillNextFrom(self.logcut(hems.TotalBroad(),-1/self.lam)*self.lam*2,w)
        self.EvtShp["WideBroad"].FillNextFrom(self.logcut(hems.WideBroad(),-1/self.lam)*self.lam*2,w)
        self.EvtShp["NarrowBroad"].FillNextFrom(self.logcut(hems.NarrowBroad(),-1/self.lam)*self.lam*2,w)
        if self.cas & 4:
            fcx = hems.FCx([0,0.5,1,1.5])
            self.EvtShp["FC0"].FillNextFrom(self.logcut(fcx[0],-1/self.lam)*self.lam*2,w)
            self.EvtShp["FC0.5"].FillNextFrom(self.logcut(fcx[1],-1/self.lam)*self.lam*2,w)
            self.EvtShp["FC1"].FillNextFrom(self.logcut(fcx[2],-1/self.lam)*self.lam*2,w)
            self.EvtShp["FC1.5"].FillNextFrom(self.logcut(fcx[3],-1/self.lam)*self.lam*2,w)
        if self.cas & 1:
            kt2c = self.camalg.Cluster(event)
            self.EvtShp["y23"].FillNextFrom(self.logcut(kt2c[0],-1/self.lam)*self.lam,w)
            y23val = self.logcut(kt2c[1][0],-1/self.lam)*self.lam
            ktrat = 0 if kt2c[1][0]==0 else kt2c[1][1]/kt2c[1][0]
            if ( y23val>-0.6 and y23val<-0.5 ) and ( ktrat>0.3**2 and ktrat<0.5**2 ):
                self.EvtShp["DPsi"].Fill(abs(kt2c[2]/m.pi),w)

    def Finalize(self,name):
        self.sn = self.n
        for h in self.ynm: h.Store()
        for h in self.EvtShp.values(): h.Store()
        self.n = MPI.COMM_WORLD.allreduce(self.n,op=MPI.SUM)
        for h in self.ynm: h.MPISync()
        for h in self.EvtShp.values(): h.MPISync()
        if MPI.COMM_WORLD.Get_rank():
            self.n = self.sn
            for h in self.ynm: h.Restore()
            for h in self.EvtShp.values(): h.Restore()
            return
        for h in self.ynm: h.ScaleW(1/self.n)
        for h in self.EvtShp.values(): h.ScaleW(1/self.n)
        if not name.endswith(".yoda"): name += ".yoda"
        file = open(name,"w")
        file.write("\n\n".join([ str(h) for h in self.ynm ]))
        file.write("\n\n".join([ str(h) for h in self.EvtShp.values() ]))
        file.close()
        self.n = self.sn
        for h in self.ynm: h.Restore()
        for h in self.EvtShp.values(): h.Restore()
