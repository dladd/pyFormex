.. $Id$  -*- rst -*-
.. pyformex reference manual --- coords
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-coords:

:mod:`coords` --- A structured collection of 3D coordinates.
============================================================

.. automodule:: coords
   :synopsis: A structured collection of 3D coordinates.



   .. autoclass:: Coords


      Coords objects have the following methods:

      .. automethod:: points()
      .. automethod:: pshape()
      .. automethod:: npoints()
      .. automethod:: x()
      .. automethod:: y()
      .. automethod:: z()
      .. automethod:: bbox()
      .. automethod:: center()
      .. automethod:: centroid()
      .. automethod:: sizes()
      .. automethod:: dsize()
      .. automethod:: bsphere()
      .. automethod:: distanceFromPlane(p,n)
      .. automethod:: distanceFromLine(p,n)
      .. automethod:: distanceFromPoint(p)
      .. automethod:: directionalSize(n,p=None)
      .. automethod:: directionalWidth(n)
      .. automethod:: directionalExtremes(n,p=None)
      .. automethod:: test(dir=0,min=None,max=None,atol=0.)
      .. automethod:: fprint(fmt="\%10.3e \%10.3e \%10.3e")
      .. automethod:: set(f)
      .. automethod:: scale(scale,dir=None,inplace=False)
      .. automethod:: translate(vector,distance=None,inplace=False)
      .. automethod:: rotate(angle,axis=2,around=None,inplace=False)
      .. automethod:: shear(dir,dir1,skew,inplace=False)
      .. automethod:: reflect(dir=0,pos=0,inplace=False)
      .. automethod:: affine(mat,vec=None,inplace=False)
      .. automethod:: cylindrical(dir=[0,1,2],scale=[1.,1.,1.],angle_spec=Deg)
      .. automethod:: hyperCylindrical(dir=[0,1,2],scale=[1.,1.,1.],rfunc=None,zfunc=None,angle_spec=Deg)
      .. automethod:: toCylindrical(dir=[0,1,2],angle_spec=Deg)
      .. automethod:: spherical(dir=[0,1,2],scale=[1.,1.,1.],angle_spec=Deg,colat=False)
      .. automethod:: superSpherical(n=1.0,e=1.0,k=0.0,dir=[0,1,2],scale=[1.,1.,1.],angle_spec=Deg,colat=False)
      .. automethod:: toSpherical(dir=[0,1,2],angle_spec=Deg)
      .. automethod:: bump1(dir,a,func,dist)
      .. automethod:: bump2(dir,a,func)
      .. automethod:: bump(dir,a,func,dist=None)
      .. automethod:: flare(xf,f,dir=[0,2],end=0,exp=1.)
      .. automethod:: newmap(func)
      .. automethod:: map(func)
      .. automethod:: map1(dir,func,x=None)
      .. automethod:: mapd(dir,func,point,dist=None)
      .. automethod:: egg(k)
      .. automethod:: replace(i,j,other=None)
      .. automethod:: swapAxes(i,j)
      .. automethod:: rollAxes(n=1)
      .. automethod:: projectOnPlane(P,n)
      .. automethod:: projectOnSphere(radius=1.,center=[0.,0.,0.])
      .. automethod:: projectOnCylinder(radius=1.,dir=0,center=[0.,0.,0.])
      .. automethod:: split()
      .. automethod:: fuse(nodesperbox=1,shift=0.5,rtol=1.e-5,atol=1.e-5,repeat=True)
      .. automethod:: append(coords)
      .. automethod:: concatenate(clas,L,axis=0)
      .. automethod:: fromstring(clas,fil,sep=' ',ndim=3,count=1)
      .. automethod:: fromfile(clas,fil)
      .. automethod:: interpolate(clas,F,G,div)

**Functions defined in the module coords**

   .. autofunction:: bbox(objects)
   .. autofunction:: coordsmethod(f)
   .. autofunction:: origin()

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

