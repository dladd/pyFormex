.. $Id$  -*- rst -*-
.. pyformex reference manual --- arraytools
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-arraytools:

:mod:`arraytools` --- A collection of numerical array utilities.
================================================================

.. automodule:: arraytools
   :synopsis: A collection of numerical array utilities.


   .. autofunction:: niceLogSize(f)
   .. autofunction:: niceNumber(f,approx=floor)
   .. autofunction:: sind(arg,angle_spec=Deg)
   .. autofunction:: cosd(arg,angle_spec=Deg)
   .. autofunction:: tand(arg,angle_spec=Deg)
   .. autofunction:: dotpr(A,B,axis=1)
   .. autofunction:: length(A,axis=1)
   .. autofunction:: normalize(A,axis=1)
   .. autofunction:: projection(A,B,axis=1)
   .. autofunction:: norm(v,n=2)
   .. autofunction:: solveMany(A,b)
   .. autofunction:: permutations(iterable,r=None)
   .. autofunction:: inside(p,mi,ma)
   .. autofunction:: isClose(values,target,rtol=1.e-5,atol=1.e-8)
   .. autofunction:: unitVector(v)
   .. autofunction:: rotationMatrix(angle,axis=None,angle_spec=Deg)
   .. autofunction:: rotMatrix(u,w=[0.,0.,1.],n=3)
   .. autofunction:: growAxis(a,size,axis=1,fill=0)
   .. autofunction:: reverseAxis(a,axis=1)
   .. autofunction:: checkArray(a,shape=None,kind=None,allow=None)
   .. autofunction:: checkArray1D(a,size=None,kind=None,allow=None)
   .. autofunction:: checkUniqueNumbers(nrs,nmin=0,nmax=None)
   .. autofunction:: readArray(file,dtype,shape,sep=' ')
   .. autofunction:: writeArray(file,array,sep=' ')
   .. autofunction:: cubicEquation(a,b,c,d)
   .. autofunction:: unique1dOrdered(ar1,return_index=False,return_inverse=False)
   .. autofunction:: renumberIndex(index)
   .. autofunction:: inverseUniqueIndex(index)
   .. autofunction:: sortByColumns(A)
   .. autofunction:: uniqueRows(A)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

