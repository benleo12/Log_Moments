
import math as m

class Vec4:

    def __init__(self,E=0,px=0,py=0,pz=0):
        self.E = E
        self.px = px
        self.py = py
        self.pz = pz

    def __getitem__(self,i):
        if i == 0: return self.E
        if i == 1: return self.px
        if i == 2: return self.py
        if i == 3: return self.pz
        raise Exception('Vec4D')

    def __setitem__(self,i,v):
        if i<0 or i>3: raise Exception('Vec4D')
        if i == 0: self.E = v
        if i == 1: self.px = v
        if i == 2: self.py = v
        if i == 3: self.pz = v

    def __repr__(self):
        return '({0},{1},{2},{3})'.format(self.E,self.px,self.py,self.pz)

    def __str__(self):
        return '({0},{1},{2},{3})'.format(self.E,self.px,self.py,self.pz)

    def __add__(self,v):
        return Vec4(self.E+v.E,self.px+v.px,self.py+v.py,self.pz+v.pz)

    def __sub__(self,v):
        return Vec4(self.E-v.E,self.px-v.px,self.py-v.py,self.pz-v.pz)

    def __neg__(self):
        return Vec4(-self.E,-self.px,-self.py,-self.pz)

    def __mul__(self,v):
        if isinstance(v,Vec4):
            return self.E*v.E-self.px*v.px-self.py*v.py-self.pz*v.pz
        return Vec4(self.E*v,self.px*v,self.py*v,self.pz*v)

    def __rmul__(self,v):
        if isinstance(v,Vec4):
            return self.E*v.E-self.px*v.px-self.py*v.py-self.pz*v.pz
        return Vec4(self.E*v,self.px*v,self.py*v,self.pz*v)

    def __div__(self,v):
        return Vec4(self.E/v,self.px/v,self.py/v,self.pz/v)

    def M2(self):
        return self*self

    def M(self):
        return m.sqrt(self.M2())

    def P2(self):
        return self.px*self.px+self.py*self.py+self.pz*self.pz

    def P(self):
        return m.sqrt(self.P2())

    def PT2(self):
        return self.px*self.px+self.py*self.py

    def PT(self):
        return m.sqrt(self.PT2())

    def Theta(self) :
        return m.acos(self.pz/self.P())

    def Phi(self) :
        if self.px==0 and self.py==0:
            return 0.0
        else:
            return m.atan2(self.py,self.px)

    def Cross(self,v):
        return Vec4(0.0,
                    self.py*v.pz-self.pz*v.py,
                    self.pz*v.px-self.px*v.pz,
                    self.px*v.py-self.py*v.px)

    def Boost(self,v):
        rsq = self.M()
        v0 = (self.E*v.E-self.px*v.px-self.py*v.py-self.pz*v.pz)/rsq;
        c1 = (v.E+v0)/(rsq+self.E)
        return Vec4(v0,
                    v.px-c1*self.px,
                    v.py-c1*self.py,
                    v.pz-c1*self.pz)

    def BoostBack(self,v):
        rsq = self.M()
        v0 = (self.E*v.E+self.px*v.px+self.py*v.py+self.pz*v.pz)/rsq;
        c1 = (v.E+v0)/(rsq+self.E)
        return Vec4(v0,
                    v.px+c1*self.px,
                    v.py+c1*self.py,
                    v.pz+c1*self.pz)

    def P3(self):
        return Vec3(self.px,self.py,self.pz)

class Vec3:

    def __init__(self,px=0,py=0,pz=0):
        self.px = px
        self.py = py
        self.pz = pz

    def __getitem__(self,i):
        if i == 1: return self.px
        if i == 2: return self.py
        if i == 3: return self.pz
        raise Exception('Vec3D')

    def __setitem__(self,i,v):
        if i<0 or i>3: raise Exception('Vec3D')
        if i == 1: self.px = v
        if i == 2: self.py = v
        if i == 3: self.pz = v

    def __repr__(self):
        return '({0},{1},{2})'.format(self.px,self.py,self.pz)

    def __str__(self):
        return '({0},{1},{2})'.format(self.px,self.py,self.pz)

    def __add__(self,v):
        return Vec3(self.px+v.px,self.py+v.py,self.pz+v.pz)

    def __sub__(self,v):
        return Vec3(self.px-v.px,self.py-v.py,self.pz-v.pz)

    def __neg__(self):
        return Vec3(-self.px,-self.py,-self.pz)

    def __mul__(self,v):
        if isinstance(v,Vec3):
            return self.px*v.px+self.py*v.py+self.pz*v.pz
        return Vec3(self.px*v,self.py*v,self.pz*v)

    def __rmul__(self,v):
        if isinstance(v,Vec3):
            return self.px*v.px+self.py*v.py+self.pz*v.pz
        return Vec3(self.px*v,self.py*v,self.pz*v)

    def __div__(self,v):
        return Vec3(self.px/v,self.py/v,self.pz/v)

    def P2(self):
        return self.px*self.px+self.py*self.py+self.pz*self.pz

    def P(self):
        return m.sqrt(self.P2())

    def PT2(self):
        return self.px*self.px+self.py*self.py

    def PT(self):
        return m.sqrt(self.PT2())

    def Theta(self) :
        return m.acos(self.pz/self.P())

    def Phi(self) :
        if self.px==0 and self.py==0:
            return 0.0
        else:
            return m.atan2(self.py,self.px)

    def Theta(self,v) :
        return m.acos(max(-1.,min(1.,(self*v)/(self.P()*v.P()))))

    def Cross(self,v):
        return Vec3(self.py*v.pz-self.pz*v.py,
                    self.pz*v.px-self.px*v.pz,
                    self.px*v.py-self.py*v.px)
