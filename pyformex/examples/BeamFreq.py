#!/usr/bin/env pyformex --gui
# $Id$

"""BeamFreq

level = 'normal'
topics = ['FEA']
techniques = ['curve','external']

.. Description

BeamFreq
--------
This example shows the first natural vibration modes of an elatic beam.

"""

from plugins.curve import *
import simple

_install="""
Calix is a free program and you can install it as follows:

- download calix (1.5-a2 or higher) from ftp://bumps.ugent.be/pub/calix/
- unpack, compile and install (as root)::

   tar xvzf calix-1.5-a2.tar.gz
   cd calix-1.5
   make
   (sudo) make install
"""

sta,out = utils.runCommand('echo stop|calix|grep -F "CALIX-"',False)
if sta:
   showText("""..

Error
-----
An error occurred when I tried to execute the program 'calix'.
This probably means that calix is not installed on your system.
""" + _install)
   exit()

s = [ si for si in out.split() if si.startswith('CALIX') ]
if (len(s) == 0) or (s[0] != 'CALIX-1.5-a2'):
   showText("""..

Error
-----
It looks like the version of 'calix' installed on your system is not
one I can use for this example.
""" + _install)
   exit()


n = 16
nshow = 4

F = simple.line([0.,0.,0.],[0.,1.,0.],n)
M = F.toMesh()

draw(M)



nnod = M.ncoords()
nel = M.nelems()
nmat = 1
iout = 1


# init
s=""";calix script written by pyFormex (example BeamFreq)
start
use program 'frame.cal'
endtext
"""
# params
s += " %s %s %s %s\n" % (nnod,nel,nmat,iout)
# nodes
for i,x in enumerate(M.coords):
   s += "%5d%10.3e%10.3e%10.3e\n" %  ((i+1,)+tuple(x))
# rvw
s += """
    1    1    1    1    1    1    1
    2    0    1    1    1    1    0%s    1

""" % (nnod-2)
# material
s += "      3.d6     1.2d6    0.0675     3000.    200.d4     70.d4    110.d4\n"
# elems
fmt = "%5s"*(M.nplex()+2) + '\n'
for i,e in enumerate(M.elems+1):
   s += fmt % ((i+1,1)+tuple(e))

# action
s += """
exec frame_ev
endtext
file open 'test.out' write seq 17
user printf '(5g13.6)' DISPL $17
file close $17
stop
"""
   
print s

fil = open('temp.dta','w')
fil.write(s)
fil.close()

cmd = "calix temp.dta temp.res"
import utils
sta,out = utils.runCommand(cmd)
#showText(out)
#showFile('temp.res')

a = fromfile('test.out',sep=' ').reshape(-1,nnod,6)
print a
print a.shape

nmodes = a.shape[0]
layout(nshow,ncols=4)
hscale = 0.5

def drawDeformed(M,u,r):
   xd = M.coords.copy()
   xd[:,0] += u
   print xd
   c = NaturalSpline(xd)
   draw(c,color=red)
   draw(c.pointsOn())

for im,ai in enumerate(a[:nshow]):
   viewport(im)
   clear()
   linewidth(2)
   draw(M)
   u = ai[:,0]
   imax = argmax(abs(u))
   r = ai[:,5]
   sc = hscale / u[imax]
   u *= sc
   r *= sc
   print u,r
   drawDeformed(M,u,r)


