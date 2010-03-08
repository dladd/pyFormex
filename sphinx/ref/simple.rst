.. $Id$  -*- rst -*-
.. pyformex reference manual --- simple
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-simple:

:mod:`simple` --- Predefined geometries with a simple shape.
============================================================

.. automodule:: simple
   :synopsis: Predefined geometries with a simple shape.


   .. autofunction:: shape(name)
   .. autofunction:: regularGrid(x0,x1,nx)
   .. autofunction:: point(x=0.,y=0.,z=0.)
   .. autofunction:: line(p1=[0.,0.,0.],p2=[1.,0.,0.],n=1)
   .. autofunction:: rect(p1=[0.,0.,0.],p2=[1.,0.,0.],nx=1,ny=1)
   .. autofunction:: rectangle(nx,ny,b=None,h=None,bias=0.,diag=None)
   .. autofunction:: circle(a1=2.,a2=0.,a3=360.,r=None,n=None,c=None)
   .. autofunction:: polygon(n)
   .. autofunction:: triangle()
   .. autofunction:: quadraticCurve(x=None,n=8)
   .. autofunction:: sphere2(nx,ny,r=1,bot=90,top=90)
   .. autofunction:: sphere3(nx,ny,r=1,bot=90,top=90)
   .. autofunction:: connectCurves(curve1,curve2,n)
   .. autofunction:: sector(r,t,nr,nt,h=0.,diag=None)
   .. autofunction:: cylinder(D,L,nt,nl,D1=None,angle=360.,bias=0.,diag=None)
   .. autofunction:: cuboid(xmin=[0.,0.,0.],xmax=[1.,1.,1.])

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

