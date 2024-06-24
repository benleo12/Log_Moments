import sys
from mymath import *
#from mpi4py import MPI

class Bin1D:

    def __init__(self,xmin,xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.w = mn(0)
        self.w2 = mn(0)
        self.wx = mn(0)
        self.wx2 = mn(0)
        self.n = mn(0)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "{0:10.6e}\t{1:10.6e}\t{2:10.6e}\t{3:10.6e}\t{4:10.6e}\t{5:10.6e}\t{6}".format\
            (self.xmin,self.xmax,float(self.w),float(self.w2),float(self.wx),float(self.wx2),float(self.n))

    def Format(self,tag):
        return "{0}\t{0}\t{1:10.6e}\t{2:10.6e}\t{3:10.6e}\t{4:10.6e}\t{5}".format\
            (tag,float(self.w),float(self.w2),float(self.wx),float(self.wx2),float(self.n))

    #def MPISync(self):
    #    comm = MPI.COMM_WORLD
    #    self.w = comm.allreduce(self.w,op=MPI.SUM)
    #    self.w2 = comm.allreduce(self.w2,op=MPI.SUM)
    #    self.wx = comm.allreduce(self.wx,op=MPI.SUM)
    #    self.wx2 = comm.allreduce(self.wx2,op=MPI.SUM)
    #    self.n = comm.allreduce(self.n,op=MPI.SUM)

    def Store(self):
        self.sw = self.w
        self.sw2 = self.w2
        self.swx = self.wx
        self.swx2 = self.wx2
        self.sn = self.n

    def Restore(self):
        self.w = self.sw
        self.w2 = self.sw2
        self.wx = self.swx
        self.wx2 = self.swx2
        self.n = self.sn

    def Width(self):
        return self.xmax-self.xmin

    def Fill(self,x,w):
        self.w += w
        self.w2 += w*w
        self.wx += w*x
        self.wx2 += w*w*x
        self.n += 1

    def ScaleW(self,scale):
        self.w *= scale
        self.w2 *= scale*scale
        self.wx *= scale
        self.wx2 *= scale*scale

class Histo1D:

    def __init__(self,nbin,xmin,xmax,name):
        self.name = name
        self.bins = []
        self.uflow = Bin1D(-sys.float_info.max,xmin)
        width = (xmax-xmin)/nbin
        for i in range(nbin):
            self.bins.append(Bin1D(xmin+i*width,xmin+(i+1)*width))
        self.oflow = Bin1D(xmax,sys.float_info.max)
        self.total = Bin1D(-sys.float_info.max,sys.float_info.max)
        self.scale = mn(1)

    def __repr__(self):
        return str(self)

    def __str__(self):
        s = "BEGIN YODA_HISTO1D {0}\n\n".format(self.name)
        s += "Path={0}\n\n".format(self.name)
        s += "ScaledBy={0}\n".format(float(self.scale))
        s += "Title=\nType=Histo1D\n"
        s += "# ID\tID\tsumw\tsumw2\tsumwx\tsumwx2\tnumEntries\n"
        s += self.total.Format("Total")+"\n"
        s += self.uflow.Format("Underflow")+"\n"
        s += self.oflow.Format("Overflow")+"\n"
        s += "# xlow\txhigh\tsumw\tsumw2\tsumwx\tsumwx2\tnumEntries\n"
        for i in self.bins: s += "{0}\n".format(i)
        s += "END YODA_HISTO1D\n"
        return s

    def Fill(self,x,w):
        l = 0
        r = len(self.bins)-1
        c = int((l+r)/2)
        a = self.bins[c].xmin
        while r-l > 1:
            if x < a: r = c
            else: l = c
            c = int((l+r)/2)
            a = self.bins[c].xmin
        if x > self.bins[r].xmin:
            if x > self.bins[r].xmax:
                self.oflow.Fill(x,w)
            else:
                self.bins[r].Fill(x,w)
        elif x < self.bins[l].xmin:
            self.uflow.Fill(x,w)
        else:
            self.bins[l].Fill(x,w)
        self.total.Fill(x,w)

    def FillTo(self,x,w):
        l = 0
        r = len(self.bins)-1
        c = int((l+r)/2)
        a = self.bins[c].xmin
        while r-l > 1:
            if x < a: r = c
            else: l = c
            c = int((l+r)/2)
            a = self.bins[c].xmin
        if x > self.bins[r].xmin:
            self.uflow.Fill(x,w)
            for i in self.bins: i.Fill(x,w)
            if x > self.bins[r].xmax:
                self.oflow.Fill(x,w)
        elif x < self.bins[l].xmin:
            self.uflow.Fill(x,w)
        else:
            self.uflow.Fill(x,w)
            for i in range(l):
                self.bins[i].Fill(x,w)
            f = x-self.bins[l].xmin
            f /= self.bins[l].Width()
            self.bins[l].Fill(x,w*f)
        self.total.Fill(x,w)

    def FillFrom(self,x,w):
        l = 0
        r = len(self.bins)-1
        c = int((l+r)/2)
        a = self.bins[c].xmin
        while r-l > 1:
            if x < a: r = c
            else: l = c
            c = int((l+r)/2)
            a = self.bins[c].xmin
        if x > self.bins[r].xmin:
            if x < self.bins[r].xmax:
                f = self.bins[r].xmax-x
                f /= self.bins[r].Width()
                self.bins[r].Fill(x,w*f)
            self.oflow.Fill(x,w)
        elif x < self.bins[l].xmin:
            self.uflow.Fill(x,w)
            for i in self.bins: i.Fill(x,w)
            self.oflow.Fill(x,w)
        else:
            f = self.bins[l].xmax-x
            f /= self.bins[l].Width()
            self.bins[l].Fill(x,w*f)
            for i in range(l+1,len(self.bins)):
                self.bins[i].Fill(x,w)
            self.oflow.Fill(x,w)
        self.total.Fill(x,w)

    def FillPreviousTo(self,x,w):
        l = 0
        r = len(self.bins)-1
        c = int((l+r)/2)
        a = self.bins[c].xmin
        while r-l > 1:
            if x < a: r = c
            else: l = c
            c = int((l+r)/2)
            a = self.bins[c].xmin
        if x > self.bins[r].xmin:
            self.uflow.Fill(x,w)
            for i in self.bins: i.Fill(x,w)
        elif x < self.bins[l].xmin:
            self.uflow.Fill(x,w)
        else:
            self.uflow.Fill(x,w)
            for i in range(l):
                self.bins[i].Fill(x,w)
        self.total.Fill(x,w)

    def FillNextFrom(self,x,w):
        l = 0
        r = len(self.bins)-1
        c = int((l+r)/2)
        a = self.bins[c].xmin
        while r-l > 1:
            if x < a: r = c
            else: l = c
            c = int((l+r)/2)
            a = self.bins[c].xmin
        if x > self.bins[r].xmin:
            self.oflow.Fill(x,w)
        elif x < self.bins[l].xmin:
            for i in self.bins: i.Fill(x,w)
            self.oflow.Fill(x,w)
        else:
            for i in range(l+1,len(self.bins)):
                self.bins[i].Fill(x,w)
            self.oflow.Fill(x,w)
        self.total.Fill(x,w)

    def ScaleW(self,scale):
        self.total.ScaleW(scale)
        self.uflow.ScaleW(scale)
        self.oflow.ScaleW(scale)
        for i in self.bins: i.ScaleW(scale)
        self.scale = scale

    #def MPISync(self):
    #    self.total.MPISync()
    #    self.uflow.MPISync()
    #    self.oflow.MPISync()
    #    for i in self.bins: i.MPISync()

    def Store(self):
        self.total.Store()
        self.uflow.Store()
        self.oflow.Store()
        for i in self.bins: i.Store()

    def Restore(self):
        self.total.Restore()
        self.uflow.Restore()
        self.oflow.Restore()
        for i in self.bins: i.Restore()


