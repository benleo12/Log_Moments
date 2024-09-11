import config
if not config.quad_precision:
    import math as m
    config.kin_epsilon = 1.e-12
    def mn(v): return float(v)
    def nm(v): return float(v)
    def myexp(a): return m.exp(a)
    def mylog(a): return m.log(a)
    def mylog10(a): return m.log10(a)
    def mypow(a,b): return m.pow(a,b)
    def mysqrt(a): return m.sqrt(a)
    def mysin(y): return m.sin(y)
    def mycos(y): return m.cos(y)
    def myacos(x): return m.acos(x)
    def myatan2(x,y): return m.atan2(x,y)
elif config.quad_precision >= 2:
    import mpmath as m
    m.mp.prec = 113*(2**(config.quad_precision-1))
    def mn(v): return m.mpf(v)
    def nm(v): return m.mpf(v)
    config.kin_epsilon = mn('1e-{0}'.format(64*(2**(config.quad_precision-2))))
    def myexp(a): return m.exp(a)
    def mylog(a): return m.log(a)
    def mylog10(a): return m.log10(a)
    def mypow(a,b): return m.power(a,b)
    def mysqrt(a): return m.sqrt(a)
    def mysin(y): return m.sin(y)
    def mycos(y): return m.cos(y)
    def myacos(x): return m.acos(x)
    def myatan2(x,y): return m.atan2(x,y)
else:
    import mpmath as mpm
    import doubledouble as m
    mpm.mp.prec = 113
    def mn(v):
        hi = float(v)
        return m.DoubleDouble(hi,mpm.mpf(v)-hi)
    def nm(v):
        return mpm.mpf(v.x)+mpm.mpf(v.y)
    config.kin_epsilon = mn('1e-28')
    def myexp(a): return a.exp()
    def mylog(a): return a.log()
    def mylog10(a): return a.log10()
    def mypow(a,b): return (a.log()*b).exp()
    def mysqrt(x): return x.sqrt()
    CCSin = [ mn('1.276278962402265880207636972086138'),
              mn('-0.2852615691810360095702940903036356'),
              mn('0.009118016006651802497767922609497572'),
              mn('-0.0001365875135419666724364765329598821'),
              mn('1.184961857661690108290062470872107e-6'),
              mn('-6.702791603827441236048382414653050e-9'),
              mn('2.667278599019659364896698962248918e-11'),
              mn('-7.872922121718594384973039392379814e-14'),
              mn('1.792294735924872672763992581936114e-16'),
              mn('-3.242712736631545645107510426112595e-19'),
              mn('4.774743247264320327618489681053052e-22'),
              mn('-5.833273647400330565578188488896630e-25'),
              mn('6.008077974456641252045947210770834e-28') ]
    def mysin(y):
        while y > m.pi*3/2: y -= 2*m.pi
        while y < -m.pi/2: y += 2*m.pi
        if y > m.pi/2: y = m.pi-y
        x, sx = 2*y/m.pi, CCSin[0]
        T = [1,x,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for i in range(2,25): T[i] = 2*x*T[i-1]-T[i-2]
        for i in range(1,13): sx += CCSin[i]*T[i+i]
        return x*sx
    def mycos(y):
        return mysin(y+m.pi/2)
    def myacos(x):
        v = mpm.acos(mpm.mpf(x.x)+mpm.mpf(x.y))
        hi = float(v)
        return m.DoubleDouble(hi,v-hi)
    def myatan2(x,y):
        v = mpm.atan2(mpm.mpf(x.x)+mpm.mpf(x.y),
                      mpm.mpf(y.x)+mpm.mpf(y.y))
        hi = float(v)
        return m.DoubleDouble(hi,v-hi)

def print_math_settings():
    if config.quad_precision >= 2:
        print(m.mp)
        print('  mp.pretty = {0}'.format(m.mp.pretty))
        print('  mp.backend = {0}'.format(m.libmp.BACKEND))
        print('  epsilon = {0}'.format(config.kin_epsilon))
    elif config.quad_precision:
        print('Using doubledouble precision')
        print(mpm.mp)
        print('  mp.pretty = {0}'.format(mpm.mp.pretty))
        print('  mp.backend = {0}'.format(mpm.libmp.BACKEND))
        print('  epsilon = {0}'.format(config.kin_epsilon))

from random import Random
from math import ldexp

class FullRandom(Random):

    def random(self):
        mantissa = 0x10000000000000 | self.getrandbits(52)
        exponent = -53
        x = 0
        while not x:
            x = self.getrandbits(32)
            exponent += x.bit_length() - 32
        return ldexp(mantissa, exponent)

rng = FullRandom()

