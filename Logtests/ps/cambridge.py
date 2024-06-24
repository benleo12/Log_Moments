from mymath import *
from vector import Vec4, Vec3, GetR, GetRinv
from particle import Particle

class Algorithm:

    def Yij(self,p,q):
        return 2*min(p.E*p.E,q.E*q.E)*p.SmallOMCT(q)/self.ecm2

    def KTij(self,p,q):
        return p.Cross(q).P2()/(p+q).P2()/self.ecm2

    def Vij(self,p,q):
        return p.SmallOMCT(q)/self.ecm2

    def Psiijk(self,pj,pk,Rn):
        delukj = pk.P3()/pk.P3().Abs()-pj.P3()/pj.P3().Abs()
        return myatan2(delukj*(Rn*Vec3(mn(0),mn(1),mn(0))),
                       delukj*(Rn*Vec3(mn(1),mn(0),mn(0))))

    def Cluster(self,event):
        self.ecm2 = (event[0].mom+event[1].mom).M2()
        p = [ i.mom for i in event[2:] ]
        idij = []
        kt2 = []
        n = len(p)
        if n == 2: return [0,[0,0],-1]
        imap = list(range(n))
        kt2ij = [ [ 0 for i in range(n) ] for j in range(n) ]
        dmin = mn('inf')
        for i in range(n):
            for j in range(i):
                dij = kt2ij[i][j] = self.Vij(p[i],p[j])
                if dij < dmin: dmin = dij; ii = i; jj = j
        yijmax = self.Yij(p[imap[ii]],p[imap[jj]])
        ktij = [self.KTij(p[imap[ii]],p[imap[jj]]),0]
        idij.append([ii,jj,imap[ii],imap[jj]])
        id23 = n-3
        while n>=3:
            n -= 1
            jjx = imap[jj]
            p[jjx] += p[imap[ii]]
            for i in range(ii,n): imap[i] = imap[i+1]
            for j in range(jj): kt2ij[jjx][imap[j]] = self.Vij(p[jjx],p[imap[j]])
            for i in range(jj+1,n): kt2ij[imap[i]][jjx] = self.Vij(p[jjx],p[imap[i]])
            dmin = mn('inf')
            for i in range(n):
                for j in range(i):
                    dij = kt2ij[imap[i]][imap[j]]
                    if dij < dmin: dmin = dij; ii = i; jj = j
            if n>2:
                idij.append([ii,jj,imap[ii],imap[jj]])
                yijmax = max([yijmax,self.Yij(p[imap[ii]],p[imap[jj]])])
                cktij = self.KTij(p[imap[ii]],p[imap[jj]])
                if cktij > ktij[1]:
                    if cktij > ktij[0]:
                        ktij[1] = ktij[0]
                        ktij[0] = cktij
                        id34 = id23
                        id23 = n-3
                    else:
                        ktij[1] = cktij
                        id34 = n-3
        if len(p) < 4: return [yijmax,[0,0],-1]
        j1 = p[imap[0]]
        j2 = p[imap[1]]
        if j1[3] > j2[3]:
            d = (j1-j2).P3()/(j1-j2).P()
            s1,s2 = 1,-1
        else:
            d = (j2-j1).P3()/(j1-j2).P()
            s1,s2 = -1,1
        R = GetR(d)
        Rinv = GetRinv(d)
        n = 0
        idij.reverse()
        while n <= max(id23,id34):
            for i in range(idij[n][0],n): imap[i+1] = imap[i]
            imap[idij[n][0]]=idij[n][2];
            pi = p[imap[idij[n][1]]]
            pk = p[imap[idij[n][0]]]
            pj = p[imap[idij[n][1]]] = pi-pk
            s = s1 if self.Vij(j1,pi) < self.Vij(j2,pi) else s2
            R = R*GetR(Rinv*s*pi.P3())
            Rinv = GetRinv(Rinv*(s*pi.P3()))*Rinv
            if n == id23: Psi23 = self.Psiijk(pj,pk,R)
            if n == id34: Psi34 = self.Psiijk(pj,pk,R)
            n += 1
        DPsi = abs(Psi34-Psi23)
        if DPsi > m.pi: DPsi = 2*m.pi-DPsi
        return [yijmax,ktij,DPsi]

from histogram import Histo1D

class Analysis:

    def __init__(self,n=1):
        self.n = 0.
        self.ynm = [[ Histo1D(100,-4.3,-0.3,'/LL_JetRates/log10_y_{0}{1}\n'.format(i+2,i+3))
                      for i in range(4) ] for i in range(n+1) ]
        self.duralg = Algorithm()

    def Analyze(self,event,w,weights=[]):
        self.n += 1.
        kt2 = self.duralg.Cluster(event)
        for j in range(len(self.ynm[0])):
            self.ynm[0][j].Fill(mylog10(kt2[-1-j]) if len(kt2)>j else -5.,w)
            for i in range(len(weights)):
                self.ynm[i+1][j].Fill(mylog10(kt2[-1-j]) if len(kt2)>j else -5.,w*weights[i])

    def Finalize(self,name):
        for h in self.ynm[0]: h.ScaleW(1./self.n)
        file = open(name+".yoda","w")
        file.write("\n\n".join([ str(h) for h in self.ynm[0] ]))
        file.close() 
        for i in range(1,len(self.ynm)):
            for h in self.ynm[i]: h.ScaleW(1./self.n)
            file = open(name+".{0}.yoda".format(i),"w")
            file.write("\n\n".join([ str(h) for h in self.ynm[i] ]))
            file.close() 

