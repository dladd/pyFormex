.. $Id$  -*- rst -*-
.. pyformex reference manual --- formex
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: <isonum.txt>
.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-formex:

:mod:`formex` --- Formex algebra in Python
==========================================

.. automodule:: formex
   :synopsis: Formex algebra in Python
   :members: vectorLength,vectorNormalize,intersectionLinesWithPlane



   .. autoclass:: Formex
      :members: element,point,coord,nelems,nplex,ndim,npoints,shape,view,getProp,maxProp,propSet,centroids,fuse,toMesh,info,asFormex,asFormexWithProp,asArray,fprint,setProp,append,select,selectNodes,points,vertices,remove,whereProp,withProp,splitProp,elbbox,unique,reverse,test,clip,cclip,mirror,centered,resized,circulize,circulize1,shrink,replic,replic2,rosette,translatem,extrude,divide,intersectionWithPlane,intersectionPointsWithPlane,intersectionLinesWithPlane,cutWithPlane,split,write,actor

**Functions defined in the module formex**

   .. autofunction:: vectorLength(vec)
   .. autofunction:: vectorNormalize(vec)
   .. autofunction:: vectorPairAreaNormals(vec1,vec2)
   .. autofunction:: vectorPairArea(vec1,vec2)
   .. autofunction:: vectorPairNormals(vec1,vec2)
   .. autofunction:: vectorTripleProduct(vec1,vec2,vec3)
   .. autofunction:: polygonNormals(x)
   .. autofunction:: pattern(s)
   .. autofunction:: mpattern(s)
   .. autofunction:: intersectionWithPlane(F,p,n)
   .. autofunction:: pointsAt(F,t)
   .. autofunction:: intersectionPointsWithPlane(F,p,n)
   .. autofunction:: intersectionLinesWithPlane(F,p,n,atol=???)
   .. autofunction:: cut2AtPlane(F,p,n,side=???,atol=???,newprops=???)
   .. autofunction:: cut3AtPlane(F,p,n,side=???,atol=???,newprops=???)
   .. autofunction:: cutElements3AtPlane(F,p,n,newprops=???,side=???,atol=???)
   .. autofunction:: connect(Flist,nodid=???,bias=???,loop=???)
   .. autofunction:: interpolate(F,G,div,swap=???)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

