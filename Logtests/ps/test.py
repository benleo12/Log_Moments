import sys
from matrix import eetojj
from durham import Analysis
from qcd import AlphaS, NC, TR, CA, CF
from shower import Shower 

alphas = AlphaS(91.1876,0.118)
hardxs = eetojj(alphas)
shower = Shower(alphas,t0=1.)
jetrat = Analysis()
r.seed(123456)
for i in range(10000):
    event, weight = hardxs.GenerateLOPoint()
    t = (event[0].mom+event[1].mom).M2()
    shower.Run(event,t)
    if not CheckEvent(event):
        print "Something went wrong:"
        for p in event:
            print p
    sys.stdout.write('\rEvent {0}'.format(i))
    sys.stdout.flush()
    jetrat.Analyze(event,weight)
jetrat.Finalize("myshower")
print ""
