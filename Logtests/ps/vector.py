import config
from mymath import *

class Vec3:

    def __init__(self,x=mn(0),y=mn(0),z=mn(0)):
        self.x = x
        self.y = y
        self.z = z

    def __getitem__(self,i):
        if i == 1: return self.x
        if i == 2: return self.y
        if i == 3: return self.z
        raise Exception('Vec3D')

    def __setitem__(self,i,v):
        if i<0 or i>2: raise Exception('Vec3D')
        if i == 0: self.x = v
        if i == 1: self.y = v
        if i == 2: self.z = v

    def __repr__(self):
        return '({0},{1},{2})'.format(self.x,self.y,self.z)

    def __str__(self):
        return '({0},{1},{2})'.format(self.x,self.y,self.z)

    def __add__(self,v):
        return Vec3(self.x+v.x,self.y+v.y,self.z+v.z)

    def __sub__(self,v):
        return Vec3(self.x-v.x,self.y-v.y,self.z-v.z)

    def __neg__(self):
        return Vec3(-self.x,-self.y,-self.z)

    def __mul__(self,v):
        if isinstance(v,Vec3):
            return self.x*v.x+self.y*v.y+self.z*v.z
        return Vec3(self.x*v,self.y*v,self.z*v)

    def __rmul__(self,v):
        if isinstance(v,Vec3):
            return self.x*v.x+self.y*v.y+self.z*v.z
        return Vec3(self.x*v,self.y*v,self.z*v)

    def __div__(self,v):
        return Vec3(self.x/v,self.y/v,self.z/v)

    def __truediv__(self,v):
        return Vec3(self.x/v,self.y/v,self.z/v)

    def Cross(self,v):
        return Vec3(self.y*v.z-self.z*v.y,
                    self.z*v.x-self.x*v.z,
                    self.x*v.y-self.y*v.x)

    def Abs2(self):
        return self*self

    def Abs(self):
        return mysqrt(self.Abs2())

    def Theta(self) :
        return myacos(self.z/self.Abs())

    def Phi(self) :
        if self.x==0 and self.y==0:
            return 0
        else:
            return myatan2(self.y,self.x)

    def CosTheta(self,v):
        mag = self.Abs2()*v.Abs2()
        if mag == 0: return 0
        return self*v/mysqrt(mag)

    def SinTheta(self,v):
        mag = self.Abs2()*v.Abs2()
        if mag == 0: return 0
        return self.Cross(v).Abs()/mysqrt(mag)

class Vec4:

    def __init__(self,E=mn(0),px=mn(0),py=mn(0),pz=mn(0)):
        self.E = E
        if isinstance(px,Vec3):
            self.px = px.x
            self.py = px.y
            self.pz = px.z
        else:
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
        if i == 0: self.E=v
        elif i == 1: self.px=v
        elif i == 2: self.py=v
        elif i == 3: self.pz=v
        else: raise Exception('Vec4D')

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

    def __truediv__(self,v):
        return Vec4(self.E/v,self.px/v,self.py/v,self.pz/v)

    def M2(self):
        return self*self

    def M(self):
        return mysqrt(self.M2())

    def P2(self):
        return self.px*self.px+self.py*self.py+self.pz*self.pz

    def P(self):
        return mysqrt(self.P2())

    def PT2(self):
        return self.px*self.px+self.py*self.py

    def PT(self):
        return mysqrt(self.PT2())

    def Theta(self) :
        return myacos(self.pz/self.P())

    def Phi(self) :
        if self.px==0 and self.py==0:
            return 0
        else:
            return myatan2(self.py,self.px)

    def P3(self):
        return Vec3(self.px,self.py,self.pz)

    def Cross(self,v):
        return Vec4(0,
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

    def SmallOMCT(self,v):
        mag = mysqrt(self.P2()*v.P2())
        if mag == 0: return 0
        pq = self.px*v.px+self.py*v.py+self.pz*v.pz
        ct = min(max(pq/mag,mn(-1)),mn(1))
        if ct < 0: return 1-ct
        st = self.Cross(v).P()/mag
        st2 = st/(2*mysqrt((1+ct)/mn(2)))
        return 2*st2**2

    def SmallMLDP(self,v):
        return self.E*v.E*self.SmallOMCT(v)


def LT(a,b,c):
    t = a[1]*b[2]*c[3]+a[2]*b[3]*c[1]+a[3]*b[1]*c[2] \
        -a[1]*b[3]*c[2]-a[3]*b[2]*c[1]-a[2]*b[1]*c[3]
    x = -a[0]*b[2]*c[3]-a[2]*b[3]*c[0]-a[3]*b[0]*c[2] \
	+a[0]*b[3]*c[2]+a[3]*b[2]*c[0]+a[2]*b[0]*c[3]
    y = -a[1]*b[0]*c[3]-a[0]*b[3]*c[1]-a[3]*b[1]*c[0] \
	+a[1]*b[3]*c[0]+a[3]*b[0]*c[1]+a[0]*b[1]*c[3]
    z = -a[1]*b[2]*c[0]-a[2]*b[0]*c[1]-a[0]*b[1]*c[2] \
	+a[1]*b[0]*c[2]+a[0]*b[2]*c[1]+a[2]*b[1]*c[0]
    return Vec4(t,-x,-y,-z)

class Rotation:
    def __init__(self,v1,v2):
        a = Vec4(0,v1.P3()/v1.P())
        b = Vec4(0,v2.P3()/v2.P())
        self.x = a
        self.y = b+a*(a*b)
        if self.y.P2() != 0:
            self.y = self.y/self.y.P()
        l = [1,2,3]
        m = [0,abs(a[1]),abs(a[2]),abs(a[3])]
        if m[l[2]] > m[l[1]]: l[1],l[2] = l[2],l[1]
        if m[l[1]] > m[l[0]]: l[0],l[1] = l[1],l[0]
        if m[l[2]] > m[l[1]]: l[1],l[2] = l[2],l[1]
        tdp = self.y[l[1]]*a[l[1]]+self.y[l[2]]*a[l[2]]
        if tdp != 0: self.y[l[0]] = -tdp/a[l[0]]
        if self.y.P2() == 0: self.y[l[1]] = 1
        self.omct = self.x.SmallOMCT(b)
        self.st = -self.y*b
    def __mul__(self,v):
        vx = -self.x*v
        vy = -self.y*v
        return v-(self.omct*vx+self.st*vy)*self.x \
            -(-self.st*vx+self.omct*vy)*self.y

class Mat3D:

    def __init__(self,vals=[[mn(1),mn(0),mn(0)],
                            [mn(0),mn(1),mn(0)],
                            [mn(0),mn(0),mn(1)]]):
        self.vals = vals

    def __getitem__(self,indt):
        if indt[0] <= 2 and indt[1] <= 2: return self.vals[indt[0]][indt[1]]
        raise Exception('Mat3D')

    def __setitem__(self,indt,v):
        if indt[0]<0 or indt[0]>2: raise Exception('Vec3D')
        if indt[1]<0 or indt[1]>2: raise Exception('Vec3D')
        self.vals[indt[0]][indt[1]] = v

    def __repr__(self):
        return '(({},{},{}),({},{},{}),({},{},{}))'.format \
            (self.vals[0][0],self.vals[0][1],self.vals[0][2],\
             self.vals[1][0],self.vals[1][1],self.vals[1][2],\
             self.vals[2][0],self.vals[2][1],self.vals[2][2])

    def __str__(self):
        return '(({},{},{}),({},{},{}),({},{},{}))'.format \
            (self.vals[0][0],self.vals[0][1],self.vals[0][2],\
             self.vals[1][0],self.vals[1][1],self.vals[1][2],\
             self.vals[2][0],self.vals[2][1],self.vals[2][2])

    def __mul__(self,v):
        if isinstance(v,Vec3):
            return Vec3(self.vals[0][0]*v.x + self.vals[0][1]*v.y + self.vals[0][2]*v.z, \
                        self.vals[1][0]*v.x + self.vals[1][1]*v.y + self.vals[1][2]*v.z, \
                        self.vals[2][0]*v.x + self.vals[2][1]*v.y + self.vals[2][2]*v.z)
        if isinstance(v,Mat3D):
            return Mat3D([[sum(x*y for x,y in zip(x_row,y_col)) \
                           for y_col in zip(*v.vals)] for x_row in self.vals])
        return Mat3D([[v*k[0],v*k[1],v*k[2]] for k in self.vals])

def GetRy(ct,st):
    return Mat3D([[ct,mn(0),st],[mn(0),mn(1),mn(0)],[-st,mn(0),ct]])

def GetRz(cp,sp):
    return Mat3D([[cp,-sp,mn(0)],[sp,cp,mn(0)],[mn(0),mn(0),mn(1)]])

def GetR(d):
    m, mt = d.Abs(), mysqrt(d[1]**2+d[2]**2)
    ct, st = d[3]/m, mt/m
    cp = mn(1) if mt == 0 else d[1]/mt
    sp = mn(0) if mt == 0 else d[2]/mt
    return GetRz(cp,sp)*GetRy(ct,st)*GetRz(cp,-sp)

def GetRinv(d):
    m, mt = d.Abs(), mysqrt(d[1]**2+d[2]**2)
    ct, st = d[3]/m, mt/m
    cp = mn(1) if mt == 0 else d[1]/mt
    sp = mn(0) if mt == 0 else d[2]/mt
    R = GetRz(cp,sp)*GetRy(ct,-st)*GetRz(cp,-sp)
    return R

