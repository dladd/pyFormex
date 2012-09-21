# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
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

"""Formian compatibility functions

This module defines some Formex methods which perform the same functionality
as the corresponding functions in `Formian`. 

The author of formex/formian had an incredible preference for newspeak:
for every concept or function, a new name was invented. While this may
give Formian the aspect of a sophisticated scientific background,
it works rather distracting and ennoying for people that are already
familiar with the basic ideas of 3D geometry, and prefer to use the
standardized terms.

Also, Formian has many different names for minor variation of the same
concept. pyFormex implements these by using extra arguments, often optional.

Finally, Formian starts numbering at one, while in pyFormex all numbering
is zero-based. That means that the y-axis is 1 in pyFormex, but 2 in Formian.

The functions defined in this module use the Formian names and conventions,
thus facilitating the transscription of Formian scripts to pyFormex.
Just import this module to make them available.

For original pyFormex scripts, the use of this module is discouraged.


Translate formian code to python

No change ::

  + - * / 
  sign (defined further)
  abs 
  sqrt,sin,cos,tan,asin,acos,atan,exp (from math)
  
  ln -> log (from math)
  ric -> int(round())
  tic -> int()
  floc -> float()
  m^n -> pow(m,n) of m**n
  f|x -> f(x)

  tran(i,j)|F -> F.translate(i-1,j)
  ref(i,j)|F  -> F.reflect(i-1,j)

"""

from formex import Formex

# methods that have equivalent in Formex

Formex.order = Formex.nelems
Formex.plexitude = Formex.nplex
Formex.grade = Formex.ndim

Formex.cantle = Formex.element
Formex.signet = Formex.point
Formex.uniple = Formex.coord
Formex.cop = Formex.remove

Formex.cantle2str = Formex.element2str
Formex.signet2str = Formex.point2str
Formex.pex = Formex.unique

# methods that can be emulated in Formex

def formex_method(f):
    setattr(Formex,f.__name__,f)
    return f

@formex_method
def give(self):
    print(self.toFormian())

@formex_method
def tran(self,dir,dist):
    return self.translate(dir-1,dist)

@formex_method
def ref(self,dir,dist):
    return self.reflect(dir-1,dist)

@formex_method
def rindle(self,n,dir,step):
    return self.replic(n,step,dir)
@formex_method
def rin(self,dir,n,dist):
    return self.replic(n,dist,dir-1)


@formex_method
def lam(self,dir,dist):
    return self+self.reflect(dir-1,dist)

@formex_method
def ros(self,i,j,x,y,n,angle):
    if (i,j) == (1,2):
        return self.rosette(n,angle,2,[x,y,0])
    elif (i,j) == (2,3):
        return self.rosette(n,angle,0,[0,x,y])
    elif (i,j) == (1,3):
        return self.rosette(n,-angle,1,[x,0,y])

@formex_method
def tranic(self,*args,**kargs):
    n = len(args)/2
    d = [ i-1 for i in args[:n] ]
    return self.translatem(*zip(d,args[n:]))
@formex_method
def tranid(self,t1,t2):
    return self.translate([t1,t2,0])
@formex_method
def tranis(self,t1,t2):
    return self.translate([t1,0,t2])
@formex_method
def tranit(self,t1,t2):
    return self.translate([0,t1,t2])
@formex_method
def tranix(self,t1,t2,t3):
    return self.translate([t1,t2,t3])

@formex_method
def tranad(self,a1,a2,b1,b2,t=None):
    return self.translate([b1-a1,b2-a2,0.],t)
@formex_method
def tranas(self,a1,a2,b1,b2,t=None):
    return self.translate([b1-a1,0.,b2-a2],t)
@formex_method
def tranat(self,a1,a2,b1,b2,t=None):
    return self.translate([0.,b1-a1,b2-a2],t)
@formex_method
def tranax(self,a1,a2,a3,b1,b2,b3,t=None):
    return self.translate([b1-a1,b2-a2,b3-a3],t)

@formex_method
def rinic(self,*args,**kargs):
    n = len(args)/3
    F = self
    for d,m,t in zip(args[:n],args[n:2*n],args[2*n:]):
        F = F.rin(d,m,t)
    return F
@formex_method
def rinid(self,n1,n2,t1,t2):
    return self.rin(1,n1,t1).rin(2,n2,t2)
@formex_method
def rinis(self,n1,n2,t1,t2):
    return self.rin(1,n1,t1).rin(3,n2,t2)
@formex_method
def rinit(self,n1,n2,t1,t2):
    return self.rin(2,n1,t1).rin(3,n2,t2)

@formex_method
def lamic(self,*args,**kargs):
    n = len(args)/2
    F = self
    for d,p in zip(args[:n],args[n:]):
        F = F.lam(d,p)
    return F
@formex_method
def lamid(self,t1,t2):
    return self.lam(1,t1).lam(2,t2)
@formex_method
def lamis(self,t1,t2):
    return self.lam(1,t1).lam(3,t2)
@formex_method
def lamit(self,t1,t2):
    return self.lam(2,t1).lam(2,t2)

@formex_method
def rosad(self,a,b,n=4,angle=90):
    return self.rosette(n,angle,2,[a,b,0])
@formex_method
def rosas(self,a,b,n=4,angle=90):
    return self.rosette(n,angle,1,[a,0,b])
@formex_method
def rosat(self,a,b,n=4,angle=90):
    return self.rosette(n,angle,0,[0,a,b])

@formex_method
def genid(self,n1,n2,t1,t2,bias=0,taper=0):
    return self.replic2(n1,n2,t1,t2,0,1,bias,taper)
@formex_method
def genis(self,n1,n2,t1,t2,bias=0,taper=0):
    return self.replic2(n1,n2,t1,t2,0,2,bias,taper)
@formex_method
def genit(self,n1,n2,t1,t2,bias=0,taper=0):
    return self.replic2(n1,n2,t1,t2,1,2,bias,taper)

@formex_method
def bb(self,b1,b2):
    return self.scale([b1,b2,1.])

@formex_method
def bc(self,b1,b2,b3):
    return self.cylindrical(scale=[b1,b2,b3])

@formex_method
def bp(self,b1,b2):
    return self.cylindrical(scale=[b1,b2,1.])

@formex_method
def bs(self,b1,b2,b3):
    return self.spherical(scale=[b1,b2,b3],colat=True)


# Some functions

def tic(f):
    return int(f)
def ric(f):
    return int(round(f))


# End
