#!/usr/bin/env pyformex --gui
# $Id$

"""BeamFreq

level = 'normal'
topics = ['FEA','curve','drawing']
techniques = ['external','viewport',]

.. Description

BeamFreq
--------
This example shows the first natural vibration modes of an elastic beam.

"""

from plugins.curve import *
import simple


_install="""
Calix is a free program and you can install it as follows:

- download calix (1.5-a5 or higher) from ftp://bumps.ugent.be/pub/calix/
- unpack, compile and install (as root)::

   tar xvzf calix-1.5-a5.tar.gz
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
if (len(s) == 0) or utils.SaneVersion(s[0]) < utils.SaneVersion('CALIX-1.5-a5'):
   showText("""..

Error
-----
It looks like the version of 'calix' installed on your system is not
one I can use for this example.
""" + _install)
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
   #print xd
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
