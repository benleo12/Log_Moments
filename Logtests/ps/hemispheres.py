from mymath import *
import random as r
from vector import Vec3,Vec4

class Hemispheres(object):

    def __init__(self, particles):
        self.moms = [p.mom for p in particles]
        self.Q = Vec4()
        for p in self.moms: self.Q += p
        self.Q2 = self.Q.M2()
        self.thrust = None
        self.axis = None
        self.hems = None
        self.broads = None
        self.masses = None

    def Thrust(self):
        if self.thrust is None:
            self.thrust, self.axis = self.find_thrust(self.moms)
        return self.thrust

    def Tau(self):
        return self.Thrust()

    def Axis(self):
        if self.axis is None:
            self.thrust, self.axis = self.find_thrust(self.moms)
        return self.axis
            
    def Hems(self):
        if self.hems is None:
            self.hems = self.hemispheres(self.moms,self.Axis())
        return self.hems

    def Masses(self):
        if self.masses is None:
            self.masses = [self.mass2(self.Hems()[0]),self.mass2(self.Hems()[1])]
        return self.masses
            
    def Broads(self):
        if self.broads is None:
            self.broads = [self.broad(self.Hems()[0],self.Axis()),
                           self.broad(self.Hems()[1],self.Axis())]
        return self.broads
            
    def HeavyMassSq(self):
        return max(self.Masses())

    def LightMassSq(self):
        return min(self.Masses())

    def SingleMassSq(self):
        return r.choice(self.Masses())

    def HeavyMinusLightMassSq(self):
        return self.HeavyMassSq()-self.LightMassSq()
    
    def TotalBroad(self):
        return sum(self.Broads())

    def WideBroad(self):
        return max(self.Broads())

    def NarrowBroad(self):
        return min(self.Broads())

    def SingleBroad(self):
        return r.choice(self.Broads())

    def FCx(self,xs):
        if not isinstance(xs,list):
            xs = [xs]
        return self.fcx(self.moms,xs,self.Axis(),self.Q2)
    
    def find_thrust(self, moms):
        if len(moms) == 2: return mn(0), Vec3()
        ms = sorted(moms, reverse=True, key=lambda mom: mom.P())
        starts = [ ms[0], ms[1] ]
        thrust = [ mn(1), Vec3() ]
        for start in starts:
            n = start.P3()/start.P()
            dist, best = 999, 1000
            while dist < best:
                best = dist
                pt = Vec3()
                nbar = Vec3()
                for mi in ms:
                    if n*mi.P3() > 0:
                        nbar += mi.P3()
                        pt += n.Cross(mi.P3())
                    else:
                        nbar -= mi.P3()
                        pt -= n.Cross(mi.P3())
                if nbar.Abs2() != 0: nbar *= 1/nbar.Abs()
                dist = pt.Abs()
                n = nbar
            tau, psum = mn(0), mn(0)
            for mi in ms:
                mip = mi.P()
                psum += mip
                omct = [ Vec4(1,n).SmallOMCT(mi),
                         Vec4(1,-n).SmallOMCT(mi) ]
                tau += mip*min(omct)
            tau /= psum
            if tau < thrust[0]:
                thrust = [ tau, n ]
        return thrust[0], thrust[1]


    def hemispheres(self, moms, axis):
        if len(moms) == 2:
            return [moms[0]], [moms[1]]
        h1 = []
        h2 = []
        for mom in moms:
            p3 = mom.P3()
            if p3*axis > 0:
                h1 += [mom]
            else:
                h2 += [mom]
        return [h1, h2]

    def mass2(self, hem):
        m2 = mn(0)
        for p in hem:
            for q in hem: m2+=p.SmallMLDP(q)
        return m2/self.Q2

    def broad(self, hem, axis):
        b = mn(0)
        for mom in hem:
            b += axis.Cross(mom.P3()).Abs()
        return b/mysqrt(self.Q2)/2

    def fcx(self, moms, x, axis, Q2, output=False):
        sumFracE = [mn(0)]*len(x)
        if len(moms) == 2:
            return sumFracE
        for i, m1 in enumerate(moms):
            for j in range(0,i):
                m2 = moms[j]
                if output:
                    print("Theta of {}.".format((m1.P3()*axis) * (m2.P3()*axis)))
                if (m1.P3()*axis) * (m2.P3()*axis) > 0:
                    E1 = m1.E
                    E2 = m2.E
                    sinTh = abs(m1.P3().SinTheta(m2.P3()))
                    omcTh = m1.SmallOMCT(m2)
                    omcTh = min([omcTh,2-omcTh])
                    if omcTh <= 0: continue
                    if output:
                        print(m1, m2)
                        print("E1 = {}, E2 = {}, cosTh = {}, sinTh = {}".format(E1, E2, cosTh, sinTh))
                    for i in range(len(x)):
                        sumFracE[i] += 2*E1*E2*pow(sinTh,x[i])*pow(omcTh,1-x[i])/Q2

        return sumFracE



if __name__=='__main__':
    phi = mn(120)/360 *2*m.pi
    print(calc_thrust([Vec4(mn(1),mn(1),mn(0),mn(0)),
                       Vec4(mn(1),mycos(phi),mysin(phi),mn(0)),
                       Vec4(mn(1),mycos(2*phi),mysin(2*phi),mn(0))], output=True))

