# *** pyformex ***
# $Id$
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 21:24:46 CET 2011)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##

"""BeamFreq

level = 'normal'
topics = ['FEA','curve','drawing']
techniques = ['external','viewport',]

.. Description

BeamFreq
--------
This example shows the first natural vibration modes of an elastic beam.
It requires an external program, calix, which can be downloaded from
ftp://bumps.ugent.be/pub/calix/
Make sure you have version 1.5-a8 or higher.

"""
from plugins.curve import *
import simple

_required_calix_version = '1.5-a8'

print utils.checkVersion('calix',_required_calix_version,True)

if utils.checkVersion('calix',_required_calix_version,True) < 0:
   showText("""..

Error
-----
An error occurred when I tried to find the program 'calix'.
This probably means that calix is not installed on your system,
or that the installed version is not one I can use for this example.

Calix is a free program and you can install it as follows:

- download calix (%s or higher) from ftp://bumps.ugent.be/pub/calix/
- unpack, compile and install (as root)::

   tar xvzf calix-%s.tar.gz
   cd calix-1.5
   make
   (sudo) make install
""" % (_required_calix_version,_required_calix_version))
   exit()

n = 16
nshow = 4
bcons = ['cantilever','simply supported']
verbose = False

res = askItems([
   ('n',n,{'text':'number of elements along beam'}),
   ('nshow',nshow,{'text':'number of natural modes to show'}),
   ('bcon',bcons[0],{'text':'beam boundary conditions','choices':bcons}),
   ('verbose',verbose,{'text':'show intermediate information'}),
   ])
if not res:
   exit()

globals().update(res)
   

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
s += " %s %s %s %s\n" % (nnod+1,nel,nmat,iout)
# nodes
for i,x in enumerate(M.coords):
   s += "%5d%10.3e%10.3e%10.3e\n" %  ((i+1,)+tuple(x))
# orientation node
s += "%5d%10.3e%10.3e%10.3e\n\n" %  (nnod+1,0.0,0.0,1.0)

# boundary conditions
s += "%5s    0    1    1    1    1    0%5s    1\n" %  (2,nnod-2)
s += "%5s    1    1    1    1    1    1\n" % (nnod+1)
if bcon == 'cantilever':
   # boundary conditions for cantilever
   s += "%5s    1    1    1    1    1    1\n" % (1)
   s += "%5s    0    1    1    1    1    0\n" % (nnod)
else:
   # boundary conditions for simply supported
   s += "%5s    1    1    1    1    1    0\n" % (1)
   s += "%5s    1    1    1    1    1    0\n" % (nnod)
s += '\n'
# material
s += "      3.d6     1.2d6      1.00     3000.      1.00     70.d4    110.d4\n"
# elems
fmt = "%5s"*(M.nplex()+3) + '\n'
for i,e in enumerate(M.elems+1):
   s += fmt % ((i+1,1)+tuple(e)+(nnod+1,))

# action and output in a format we can easily read back
s += """
exec frame_ev
endtext
intvar name nnod 1
intvar name ndof 7
file open 'test.out' write seq 17
user printf '(i5)' nnod $17
user printf '(i5)' ndof $17
user printf '(5g13.4)' EIG $17
user printf '(5g13.4)' DISPL $17
file close $17
stop
"""
   
#print s

fil = open('temp.dta','w')
fil.write(s)
fil.close()


if verbose:
   # show calix input data
   showFile('temp.dta')

# run calix
cmd = "calix temp.dta temp.res"
if os.path.exists('test.out'):
   os.remove('test.out')
 
sta,out = utils.runCommand(cmd)

if verbose:
   # show calix output
   showText(out)
   showFile('temp.res')
   showFile('test.out')

# read results from eigenvalue analysis
fil = open('test.out','r')
nnod,ndof = fromfile(fil,sep=' ',count=2,dtype=int)
eig = fromfile(fil,sep=' ',count=4*ndof).reshape(ndof,4)

nshow = min(nshow,ndof)
freq = eig[:nshow,2]
basefreq = freq[0]
print "Frequencies: %s" % freq
print "Multipliers: %s" % (freq/freq[0])

a = fromfile(fil,sep=' ',).reshape(-1,nnod,6)
#print a.shape
# remove the extra node
a = a[:,:-1,:]
#print a.shape

layout(nshow,ncols=4)
hscale = 0.5

def drawDeformed(M,u,r):
   xd = M.coords.copy()
   xd[:,0] += u
   c = NaturalSpline(xd)
   draw(c,color=red)
   draw(c.pointsOn())

for i in range(nshow):
   viewport(i)
   clear()
   transparent(False)
   lights(False)
   linewidth(2)
   draw(M)
   ai = a[i]
   u = ai[:,0]
   imax = argmax(abs(u))
   r = ai[:,5]
   sc = hscale / u[imax]
   u *= sc
   r *= sc
   #print u,r
   drawDeformed(M,u,r)
   fi = freq[i]
   mi = fi/freq[0]
   drawText('%s Hz = %.2f f0' % (fi,mi),20,20,size=20) 

# End
